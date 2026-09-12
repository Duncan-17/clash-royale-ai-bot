# Clash Royale AI — Learning Data Schema (Phase 7 → 9)

Status: **IMPLEMENTED (v1) 2026-09-12** in `data/learning_logger.py` (wired into
`game_knowledge/tv_royale_analyzer.py` + `data/tv_royale_watcher.py --learn`). Writes
`data/learning/{games.jsonl, samples.jsonl, frames/<game>/*.jpg}`. Decisions taken
(Duncan): egocentric self/opp with opp PRIVILEGED; per-play samples (no-op ticks
later); position DEFERRED but a confidence-gated best-effort placement is logged when
unambiguous; board via the detector (transfer to TV Royale still to validate live);
raw frame archived per play + end screen. GAP: match RESULT (crowns) -- the end frame
is archived, but parsing crowns needs an end-screen region calibration (a capture from
Duncan), so `result` is null for now.

CROSS-EDUCATION wired (the three systems teach each other): (#1) own placement is free →
only ENEMY `action.position` is perceived; (#2) a fully-known deck soft-restricts the hand
classifier to its 8 cards; (#3) decks + play events resolve board SIDE (resolve_sides:
bar[b] > owner[o] > deck[d] > position[p], struggle "!" for off-deck troops). NEXT PROBLEM
(new chat) is BOARD UNIT TRACKING — a play↔unit binding so `board.units` get card+side from
the play (fixes shared-card mis-side + class flicker) and enemy `action.position` becomes
exact; plus a struggle-crop review queue to retrain the detector. See Project_Status
">>> HANDOFF: BOARD UNIT TRACKING + STRUGGLE-CROP QUEUE <<<". Original draft below.

Draft (2026-09-11). Companion to
`clash-royale-strategy-philosophy_1.md` (the *why*) and `clash-royale-ai-roadmap.md`
Phases 7–9 (imitation → reward → RL). This doc defines the on-disk record format the
learning sits on: what we log from every play, in what shape, and — crucially — with
honest tags for what's *reliable*, what's *privileged*, and what's *not available yet*.

The plays log the TV Royale engine writes today (`{game, side, card, cost, t}`) is the
seed of this; this schema is its principled expansion.

---

## 0. What this is / isn't

- **Is:** the *data contract* — the fields, types, and semantics of one training
  sample, plus the per-game header. Written to be reviewed and argued with before any
  extractor is built.
- **Isn't:** the model, the encoder (one-hot/normalization/tensorization is a
  downstream dataloader concern — we log **raw, human-readable** values), or the
  extraction code. Those come after this format is agreed.

---

## 1. Design principles

These are the non-obvious commitments; the rest of the doc follows from them.

### 1.1 Egocentric canonicalization — every sample is from the actor's point of view
A single policy always acts as "me." So every sample is logged from the **acting
player's** perspective, regardless of whether that was the top or bottom player in the
replay:
- `self_*` = the actor's own side (hand, elixir, cycle). `opp_*` = the other player.
- Spatial fields (`board`, unit positions, tower slots) are stored in a **canonical
  frame where "self" always defends the bottom and attacks upward.** When the actor was
  the **top** player, the board is mirrored (`y → H − y`) and the detector's
  colour-coded side is flipped so `ally`=self. `mirrored: true` records this so the raw
  frame is recoverable.
- Why: both players' plays then land in the *same* frame, so top- and bottom-player
  data train one policy — and it unifies cleanly with self-play (the bot is always
  "self"), our cleaner long-term source.

> Detector side note for the extractor: in TV Royale the game colours the **bottom**
> player blue (`ally`) and **top** player red (`enemy`) from the spectator's fixed view.
> So actor-relative `self` = the actor's own colour: bottom actor → detector-`ally`=self;
> top actor → detector-`ally`=opp (then mirror).

### 1.2 Observable vs. privileged — protect live transfer
TV Royale shows the spectator **both** hands and **both** elixir bars. A real player
sees only their **own**. If we feed opponent hand/elixir straight into the policy, it
learns to depend on information it will never have live — a train/serve gap that breaks
on the real client. So every field is tagged:
- **`observable`** — available to the actor in a real game (own hand/elixir/cycle;
  the board; towers). These are the **policy's inputs**, and they transfer live.
- **`privileged`** — visible to us only because we're spectating (opp exact hand, opp
  exact elixir, opp full deck, the match result). We **log them** (they're gold), but
  they feed only: (a) the reward function, (b) a **belief/opponent model** trained to
  *predict* them from observables (philosophy §2), and (c) a privileged critic in RL.
  **Never a direct input to the live policy.**

### 1.3 Exact vs. inferred — carry the noise honestly
Reconciles the 2026-09-10 direction ("don't hardcode brittle vision-derived state").
That warning was about building deterministic *control rules* on noisy detections. Here
we do the opposite: we expose observations as **features** and tag their reliability, so
the *model* learns to handle noise rather than us pretending it away.
- **`exact`** — own elixir, own hand, own cycle, per-card costs, the decks (from the
  menu). Ground truth.
- **`inferred`** — anything from the detector: units, positions, HP%, levels, and
  (later) the board-localized play position. Logged with a confidence where we have one.

### 1.4 Source-agnostic
One record format for **both** data sources, distinguished by a `source` field:
- `"tv_royale"` — real strong-player games, both perspectives, imitation **seed**.
- `"self_play"` — the bot's own plays (own-play ground truth via `play_tracker`), the
  cleaner long-term source that comes online as the bot plays.

### 1.5 Traceable to the philosophy
Every field maps to a philosophy tag (§6). If a field can't be traced to a
`[STATE FEATURE]`, `[ALGORITHMIC MODULE]`, or `[REWARD TERM]`, it doesn't belong here.

---

## 2. File layout — two normalized JSONL files

Per-game context is written once, not repeated on every play.

| File | One row per | Purpose |
|---|---|---|
| `data/learning/games.jsonl` | game/replay | header: decks, arena, result, source |
| `data/learning/samples.jsonl` | play (later: also no-op tick) | the `state → action` samples, each referencing `game_id` |

(Keeps the current throwaway validation log and the raw `tv_royale_plays_log.jsonl`
separate — those stay as-is; this is the *curated* learning dataset built from them.)

---

## 3. Game header record (`games.jsonl`)

```jsonc
{
  "game_id": "a1b2c3",            // stable id (deck-signature hash, as today)
  "source": "tv_royale",         // or "self_play"
  "arena": 15,                    // if known from the list; else null
  "t_start_unix": 1788808182.34, // wall clock of match start (for ordering only)
  "decks": {                      // EXACT from the TV Royale menu (privileged live,
    "self": ["hog_rider", ...],   //   but a live player learns their own deck; the
    "opp":  ["golem", ...]        //   OPP deck is privileged until revealed — see §4)
  },
  "result": {                     // PRIVILEGED. null until we read the end screen (§5 gap)
    "self_crowns": 2, "opp_crowns": 1, "outcome": "win"  // win|loss|draw
  },
  "notes": ""                     // free text (e.g. "sides validated", detector caveats)
}
```

Note: because samples are egocentric, **one replay yields two logical "games"** — one
per actor perspective (self=bottom, self=top-mirrored) — or, equivalently, one game_id
with `actor_side` on each sample. Recommended: **one game_id per replay**, `actor_side`
per sample, and `decks.self/opp` resolved per-sample by the actor. (Open decision 9.1.)

---

## 4. Sample record (`samples.jsonl`) — the core

One row = the state at the instant of a play + the action taken. Grouped by tag.

```jsonc
{
  // ---- identity / sequence -------------------------------------------------
  "game_id": "a1b2c3",
  "seq": 42,                 // monotonic within game (ordering for sequence models)
  "t": 63.5,                 // MATCH-RELATIVE seconds (not wall clock) — see §5 fix
  "actor_side": "friendly",  // which player acted ("friendly"=bottom, "enemy"=top)
  "mirrored": false,         // true if actor was top → board was y-flipped (§1.1)
  "source": "tv_royale",
  "type": "play",            // "play" now; "tick" reserved for no-op sampling (§9.2)

  // ---- ACTION (the label) --------------------------------------------------
  "action": {
    "card": "hog_rider",     // [OBSERVABLE][EXACT] the played card slug
    "hand_slot": 2,          // 0–3, which hand position it came from
    "deck_index": 5,         // 0–7 within self deck (stable card identity)
    "cost": 4,               // [ALGORITHMIC] from config/cards.json
    "position": null,        // {x,y} normalized 0–1 in canonical frame.
                             //   [INFERRED] DEFERRED — null until play-localization (§5)
    "phase": "single"        // elixir phase at action time: single|double|triple
  },

  // ---- SELF (observable, exact) — the policy's core inputs ------------------
  "self": {
    "elixir": 7.3,                       // [OBSERVABLE][EXACT] continuous 0–10
    "leaked": false,                     // [REWARD/STATE] at 10 while regenerating (§4.1)
    "hand": ["hog_rider","ice_spirit","musketeer","fireball"], // 4 slugs, slot order
    "hand_costs": [4,1,4,4],             // [ALGORITHMIC]
    "affordable": ["ice_spirit","hog_rider"], // [ALGORITHMIC] cost<=elixir, cheapest-first
    "cycle": {                           // [ALGORITHMIC] own-deck cycle (known)
      "on_deck": "skeletons",            //   next card to draw
      "rest_of_queue": ["giant","zap","musketeer"],
      "fully_resolved": true
    }
  },

  // ---- BOARD + TOWERS (observable, inferred/noisy) -------------------------
  // Canonical frame: self defends bottom. Positions normalized 0–1.
  "board": {                             // [OBSERVABLE][INFERRED] from detector+thin_tracker
    "units": [
      {"type":"giant","side":"enemy","x":0.34,"y":0.28,"hp_frac":0.81,
       "level":11,"conf":0.93},
      {"type":"musketeer","side":"ally","x":0.5,"y":0.72,"hp_frac":1.0,
       "level":null,"conf":0.88}
    ],
    "detector_ok": true                  // false/absent if detector-on-TVR unvalidated
  },
  "towers": {                            // [OBSERVABLE][INFERRED] tower_reader HP fill in
    "self_left":  {"hp_frac":1.0,"alive":true},   //   detector-located boxes (RULE 2)
    "self_king":  {"hp_frac":1.0,"alive":true},
    "self_right": {"hp_frac":0.62,"alive":true},
    "opp_left":   {"hp_frac":0.0,"alive":false},
    "opp_king":   {"hp_frac":1.0,"alive":true},
    "opp_right":  {"hp_frac":0.9,"alive":true}
  },

  // ---- OPPONENT (PRIVILEGED — never a live-policy input) --------------------
  "opp": {
    "elixir": 5.1,                       // [PRIVILEGED] exact (live: a belief §2)
    "hand": ["golem","baby_dragon","lightning","tornado"], // [PRIVILEGED] exact
    "cycle": {"on_deck":"elixir_collector","rest_of_queue":["...","...","..."],
              "fully_resolved": true},
    "deck_revealed": ["golem","baby_dragon","lightning"]  // [OBSERVABLE-ISH] cards a live
                                          //   player WOULD have seen by now — the belief
                                          //   model's supervised target vs opp.hand/deck
  },

  // ---- REWARD-RAW channels (log now; reward fn computed offline in Phase 8) --
  "reward_raw": {                        // deltas since previous sample for this actor
    "elixir_spent": 4,                   // [§5.1] this action's cost
    "self_tower_hp_delta": 0.0,          // [§5.1 tower-damage channel]
    "opp_tower_hp_delta": -0.11,         // negative = we damaged them (good)
    "leaked_since_prev": 0.0             // [§4.1] elixir wasted at cap since last sample
    // trade value (unit-death elixir differential, §5.1) is DERIVED later from the
    // board/elixir time-series, not logged per-sample yet — see §5.
  }
}
```

---

## 5. Availability tiers — what fills today vs. the gaps

| Field group | NOW (built) | SOON (small add) | LATER |
|---|---|---|---|
| action.card / hand_slot / cost | ✅ dual cycle tracker + costs | | |
| action.position | | | ⛔ play-localization (deferred; philosophy §5.2, roadmap Phase 7 frame-diff, rebuilt thin + confidence-gated) |
| action.phase | ⚠️ derive from match-relative `t` (needs reliable t0) | game-clock reader for exactness | |
| self.elixir / hand / costs / affordable / cycle | ✅ elixir bar + classifier + cycle_tracker + cards.json | | |
| board.units | ⚠️ detector on TVR — **assumption to validate** (handoff) | confirm/calibrate detector-on-TVR | position-tracking quality |
| towers | ✅ tower_reader HP in detected boxes | | |
| opp.elixir / hand / cycle / deck_revealed | ✅ spectator both-hands + menu decks | | belief-model target use |
| reward_raw.elixir_spent / tower deltas / leaked | ✅ costs + tower_reader + elixir over time | | |
| reward_raw.trade_value | | | ⛔ needs unit-death tracking (Phase 8 v2) |
| result (win/loss) | ⛔ **not read** — need to OCR/crown-read the end screen | small perception add | |
| `t` match-relative | ⚠️ **fix**: log t−t0, not wall clock (current logger uses wall clock) | | |

Two concrete near-term perception adds fall out of this: **(a)** read the **match
result** (crowns) off the end screen — needed for the sparse win/loss reward; **(b)**
confirm **detector-on-TVR** so `board`/`towers` are trustworthy (already on the
validation list).

---

## 6. Mapping to the philosophy tags

| Philosophy § | Concept | Schema field(s) | Tag | Consumer |
|---|---|---|---|---|
| §0/§1 | costs, elixir, phase | `self.elixir`, `hand_costs`, `action.phase` | ALGORITHMIC / STATE | policy |
| §2.1 | own cycle | `self.cycle` | ALGORITHMIC | policy |
| §2.2/2.3 | opp deck/hand belief | `opp.*`, `opp.deck_revealed` | PRIVILEGED (target) | belief model |
| §3 | cycle/elixir analysis | derived from `self.cycle` + costs | ALGORITHMIC | policy/features |
| §4.1 | leaked elixir | `self.leaked`, `reward_raw.leaked_since_prev` | REWARD + STATE | reward, policy |
| §4.3 | banked value (units on board) | `board.units` (+ hp/pos/count) | STATE (HYBRID consumes) | policy, value model |
| §5.1 | trade value | `reward_raw.*tower_hp_delta`, `elixir_spent`, later trade_value | REWARD | reward |
| §5.2 | positional value | `board.units` positions, `action.position` | STATE / LEARNED | policy (later) |
| §5.3 | win/loss (sparse) | `games.result` | REWARD | RL |
| §6 | matchups/interactions | `board.units` types+levels (matchup table is separate algorithmic module) | STATE | policy |
| §7 | archetype | `games.decks` (classifier is a separate module) | HYBRID input | reward-scaling, plan |

---

## 7. Who consumes what

- **Imitation policy (Phase 7):** trains `p(action | observable state)` — inputs =
  `self.*` + `board` + `towers` + `action.phase` (+ `opp.deck_revealed` as the
  belief-ish input, **not** `opp.hand/elixir`). Label = `action.card` (and later
  `action.position`).
- **Belief / opponent model (philosophy §2):** trains to predict `opp.hand` / `opp.deck`
  from observables — `opp.deck_revealed` is the running target; `opp.hand`/`opp.cycle`
  are the full-supervision targets available only from TVR.
- **Reward function (Phase 8) & RL (Phase 9):** consume `reward_raw.*` + `games.result`;
  a privileged critic may additionally read `opp.*`.

---

## 8. Schema as code (for the future extractor to target)

```python
# proposed: data/learning/schema.py  (dataclasses; serialize to the JSONL above)
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Unit:
    type: str; side: str            # side: "ally" | "enemy" (actor-relative, canonical)
    x: float; y: float              # normalized 0–1, canonical frame (self = bottom)
    hp_frac: Optional[float] = None
    level: Optional[int] = None
    conf: Optional[float] = None

@dataclass
class Action:
    card: str; hand_slot: int; deck_index: int; cost: Optional[int]
    phase: str = "single"                        # single|double|triple
    position: Optional[dict] = None              # {"x":..,"y":..} or None (deferred)

@dataclass
class Sample:
    game_id: str; seq: int; t: float             # t = match-relative seconds
    actor_side: str; mirrored: bool; source: str
    type: str                                    # "play" | "tick"
    action: Action
    self_state: dict                             # elixir, leaked, hand, hand_costs, affordable, cycle
    board: dict                                  # {"units":[Unit...], "detector_ok":bool}
    towers: dict                                 # 6 slots -> {"hp_frac","alive"}
    opp: dict                                    # PRIVILEGED: elixir, hand, cycle, deck_revealed
    reward_raw: dict

@dataclass
class GameHeader:
    game_id: str; source: str
    decks: dict                                  # {"self":[...], "opp":[...]}
    arena: Optional[int] = None
    t_start_unix: Optional[float] = None
    result: Optional[dict] = None                # PRIVILEGED, may be null
    notes: str = ""
```

---

## 9. Open decisions for Duncan

1. **One game_id per replay with `actor_side` per sample** (recommended), vs. splitting
   each replay into two egocentric "games." One id + per-sample actor is simpler and
   keeps both perspectives explicitly paired.
2. **Timing / "when to wait":** log **plays only** now (teaches card *selection*), and
   add **periodic no-op `tick` samples** later when we tackle *when* to play (behavior
   cloning can't learn "wait" from play-only data — it needs negatives). Agree to design
   the format for it now (the `type` field) but not collect ticks yet?
3. **Privileged split as stated** — confirm the policy must stay live-transferable, so
   `opp.hand`/`opp.elixir` are privileged (reward/belief only), never policy inputs.
4. **Match result:** worth the small perception add to read end-screen crowns now (for
   sparse win/loss reward), or defer until Phase 8/9?
5. **Position:** confirm deferring `action.position` (null now), added when
   play-localization is rebuilt.

---

## 10. Next steps (once this is agreed)

1. Fix the live logger to write **match-relative `t`** + keep `confirmed`/`desync`
   (the two fixes already flagged from the validation code-read).
2. Add the state snapshot to `TvrAnalyzer.observe()` at the play frame — it already has
   the frame, both hands, both elixir, and (optionally) the detector — to populate a
   `Sample`. Write `games.jsonl` + `samples.jsonl`.
3. (Small perception adds) end-screen crown read; confirm detector-on-TVR for `board`.
4. Then Phase 7: an offline dataloader (encode raw → tensors) + a first behavior-cloning
   policy on `action.card`.
