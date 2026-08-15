# Clash Royale AI — Strategic Philosophy & Implementation Spec

Author: Dunk (strong CR player). This is the source-of-truth for *why* the bot values what it values. Every reward term, state feature, and submodule in the codebase should trace back to something in this doc.

## Legend

- **[ALGORITHMIC MODULE]** — deterministic, hand-coded, no learning needed. The numbers exist in the game's data, so compute directly rather than hoping a model infers it.
- **[STATE FEATURE]** — computed and fed into the model as an input. Doesn't dictate what the model does with it, just guarantees it's visible.
- **[REWARD TERM]** — a value judgment; becomes a term in the reward function.
- **[HYBRID: SEEDED + LEARNED]** — starts from a hand-designed base structure (your expertise), but the system is free to refine, reweight, or subdivide beyond that structure as it sees more data. Used wherever you don't want a fully hard-coded rule but also don't want to start from nothing.
- **[LEARNED / EMERGENT]** — not hand-coded at all. Too context-dependent to formalize reliably; the state features and reward terms around it should be good enough that the model discovers it on its own.

---

## 0. Game Constants

Two different kinds of data here, sourced differently:

- **Card roster, elixir costs, rarities** — `[ALGORITHMIC MODULE]`, fetched at runtime/launch from a Clash Royale data API rather than hardcoded, so balance patches don't silently break the bot. The official Supercell API (developer.clashroyale.com) covers roster/elixir-cost/rarity metadata but does **not** reliably expose full combat stats (HP, damage, hit speed, range) — that gap is why community sites like deckshop.pro and RoyaleAPI's data mirrors exist. Plan on pulling roster/cost data from the official API and full combat stats from a community-maintained source, cached locally at startup with a periodic refresh.
- **Elixir regen timing** (1/2.8s normal, 1/1.4s double, 1/0.9s triple, phase boundaries at 2:00 and in overtime) — this is a fixed client rule, not typically exposed through any API. Keep it as a config constant, spot-checked against the live client periodically rather than fetched at runtime.

---

## 1. Elixir Fundamentals

`[STATE FEATURE]` + `[ALGORITHMIC MODULE]` — current elixir, game phase (drives regen rate), per-card cost. All exact and known; pure plumbing.

---

## 2. Card Cycle & Opponent Modeling

The game shows you only your own hand, never the opponent's — so opponent hand/deck knowledge is inference, not observation.

1. Own deck/cycle position — `[ALGORITHMIC MODULE]`, fully known.
2. Opponent's full 8-card deck — `[STATE FEATURE]` as a belief distribution until all 8 have been seen (prior informed by meta/archetype data), `[ALGORITHMIC MODULE]` once fully known.
3. Opponent's current hand — deterministic once the full deck and cycle position are known (`[ALGORITHMIC MODULE]`); probabilistic `[STATE FEATURE]` before that.

---

## 3. Cycle + Elixir Analysis

`[ALGORITHMIC MODULE]`. Once the opponent's deck is known: average/median/min/max deck cost, and **cycle count** — elixir/cards required before a specific card comes back around (the reason low-average-cost decks like hog cycle can loop their win condition fast). Track this in both directions: how soon their key card returns, and how soon yours does.

---

## 4. Elixir Management

1. **Leaked elixir** (sitting at 10 while regenerating) — `[REWARD TERM]` + `[STATE FEATURE]`, unambiguous penalty.
2. **Overcommitment / patience** — `[LEARNED / EMERGENT]`. Don't reward "not spending elixir" directly — that's a reward-hacking trap (a bot that never acts technically maximizes it). Reward the downstream result (better trades, punished overcommitment in §5) heavily enough that patience emerges as the strategy that gets there.
3. **Two elixir scores** (bar elixir vs. bar + board-invested elixir) — `[STATE FEATURE]`, via a `card_banked_value(card, health, position, count)` module. **Settled design:** seed with a base framework of hand-designed archetype categories (tank, swarm, building, spell, support, win-condition, etc.), each with a starting value-curve template — but the actual mapping from (category, health%, position, unit count) → banked value is a learned function, not a fixed formula, conditioned on the category. This lets the system refine value curves per sub-case (e.g. distinguishing "small swarm" from "flying swarm") as it sees more decks, rather than being capped at whatever categories you enumerated up front. `[HYBRID: SEEDED + LEARNED]`.

---

## 5. Card Value

1. **Trade value** — base signal is elixir differential + a separate tower-damage channel, `[REWARD TERM]`. **Settled design:** this gets scaled by a win-path-importance multiplier — how central the traded card is to the opponent's (or your own) win condition. Killing a Hog Rider for free against a hog-cycle deck is worth more than an equivalent-elixir trade against a deck where that card isn't the win condition. The importance weighting comes from the archetype/win-path analysis in §7, using the same seed-and-refine pattern as §4.3: start from your own hand-curated importance weights per known archetype, let the system refine those weights from observed trade outcomes over time. `[HYBRID: SEEDED + LEARNED]`. Dependency note: this scaling layer needs §7's archetype classification online first — ship plain elixir differential initially, layer in the win-path multiplier once §7 exists.
2. **Positional value** (blocking a lock-on, saving a more important card, otherwise shifting an interaction) — `[LEARNED / EMERGENT]`. Expose rich enough state features (targeting state, positions, what's locked onto what) and let the network find this from outcomes rather than hand-deriving it.
3. **Trade-value-dominant reward weighting** — cumulative trade value should be the dense, frequently-firing reward signal; win/loss stays a sparse top-level signal on top of it, not the primary one. `[REWARD TERM]`.

---

## 6. Card Interactions

1. **1v1 head-to-head matchups** — `[ALGORITHMIC MODULE]`. Card stats are exact public numbers; build a matchup table / combat calculator rather than learning arithmetic from pixels.
2. **Group interactions** (target priority, tank vs. backline) — targeting rules themselves are fixed and known (`[ALGORITHMIC MODULE]`); predicting the emergent outcome of a multi-unit scrum is `[LEARNED / EMERGENT]`, built on top of those rules as input features.
3. **Must-maintain-or-exploit interactions** (e.g., keep anti-air available while Balloon is in the opponent's hand) — not a new module; this is §2's cycle/hand tracking combined with §6.1's matchup table, applied together.

---

## 7. Deck Archetype Comparison

**Settled design:** same seed-and-refine pattern as §4.3. Start from a base taxonomy of known archetypes (beatdown/golem, cycle/hog, siege/X-Bow, bait, etc.), informed by your meta knowledge and RoyaleAPI deck data — but let the classifier form its own finer-grained or custom sub-archetypes as it sees decks that don't cleanly fit the base categories, rather than forcing every deck into a fixed bucket. `[HYBRID: SEEDED + LEARNED]`. This module's output feeds two consumers: the win-path context for §5.1's trade-value scaling, and the overall game-plan layer in §8.

---

## 8. Path to Victory / Game Plan

Synthesizes archetype + card economy + key interactions into an overarching plan, while still respecting the tactical fundamentals above. Build this last — after the tactical layer (elixir management, card value, card interactions) is already making sound decisions on its own. Early on, treat the game plan as a soft `[STATE FEATURE]` (archetype + win-condition label) feeding the same tactical decision-maker, rather than a separate hard-coded planning layer; promote it to a real hierarchical structure later only if the tactical layer demonstrably needs one.

---

## Suggested build order

| Order | Module | Source | Type |
|---|---|---|---|
| 1 | Runtime data fetch (roster/costs from official API, combat stats from community source) | §0 | Algorithmic |
| 2 | Elixir regen tracking | §1, §0 | Algorithmic |
| 3 | Card matchup table (1v1 combat calc) | §6.1 | Algorithmic |
| 4 | Own/opponent cycle tracker + belief state | §2, §3 | Algorithmic + state feature |
| 5 | Deck archetype classifier (seeded taxonomy) | §7 | Hybrid |
| 6 | Card banked-value curves (seeded by category) | §4.3 | Hybrid |
| 7 | Full state representation assembly | all above | State feature |
| 8 | Reward function v1: leak penalty, plain elixir differential, tower damage, win/loss | §4.1, §5.1, §5.3 | Reward |
| 9 | Reward function v2: win-path-weighted trade value (needs §7 online) | §5.1 | Hybrid |
| 10 | Positional/group-interaction value, game-plan layer | §5.2, §6.2, §8 | Learned/emergent — built on top of 1–9, not before |

Rows 1–6 are independently buildable and testable before any RL training starts.


CRAI_Proj/
├── clash-royale-ai-roadmap.md          (already there)
├── clash-royale-strategy-philosophy_1.md  (already there)
├── README.md                            (setup + run instructions)
├── requirements.txt
├── test_connection.py                   (run this first)
├── config/settings.py                   (ADB host/port, LDPlayer path)
├── control/actions.py                   (tap/swipe wrapper)
├── perception/capture.py                (screenshot capture)
└── vendor/                              (empty, for reference repos)
