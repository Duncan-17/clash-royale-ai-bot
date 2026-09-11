# CRAI — Timeline & Time Estimates

_Written 2026-09-07, updated 2026-09-08. Read `Project_Status` for live detail._
Effort = focused working days; Calendar = intermittent evenings. Stage D depends
on a reliable detector but can be designed anytime.

## Where we are (2026-09-11) — read Project_Status "HANDOFF" block
Perception DONE (our YOLOv8l detector v4 default; thin_tracker; play_tracker with
own-play ground truth). Rule-based baseline BOT plays live (game_knowledge/bot.py,
--live validated). Hand-card CLASSIFIER (YOLOv8-cls, assets/models/hand_cls) reads
TV Royale hands ~100%. TV ROYALE ANALYSIS ENGINE wired: dual cycle tracker → both
players' plays (data/tv_royale_plays_log.jsonl), self-stocking review queues for
uncertain cards + unknown troop levels, in the unattended watcher. **NEXT (new
chat): the SELF-LEARNING logic** (roadmap Phases 7–9: imitation from the plays →
reward → RL). Everything it needs is built; see Project_Status handoff.

## Where we are (2026-09-08)
Perception primitives (hand reading, elixir, towers, unit tracking) — **built**.
TV Royale collection macro — **working**. Our own YOLOv8l detector — **trained**
(v1 synthetic → v2 +components → v3 synthetic+real, in flight). Live-tested: it
out-recalls KataCR and reads side/spawn/level natively. Now in a **detector-
iteration + REDUCE** phase: strip the compensation stack built for the old weak
detector, judge the new detector raw (`perception/detector_view.py`), fix the
detector's real weaknesses (archers = domain gap → real data), then rebuild a
thin tracker.

## Stage A — Our fine-tuned detector ✅ DONE (iterating)
| Step | State |
|---|---|
| A1 | jax-free synthetic generator (`generate_dataset.py`) — **done** (scale, playfield, components) |
| A2 | Datasets generated; real frames converted (`prepare_real_data.py`) — **done** |
| A3 | Trained v1/v2/v3 on the 5080 — **done** (v3 overnight) |
| A4 | Wrapped (`yolo_detector.py`), evaluated live vs KataCR (`detector_view.py`) — **done** |

**Milestone hit: a working, retrainable detector that's ours, better recall than
KataCR.** Remaining detector work is iteration: v3 (real data) to fix archers +
animation losses, then promote and move on. Data quality (real > synthetic for
subtle units) is the main variable.

## Stage A.5 — Thin tracker + detector polish ← CURRENT
| Step | Work | Effort |
|---|---|---|
| Promote v3 to `DEFAULT_MODEL` | done alongside this | mins |
| **Thin tracker** — identity + class debounce (fixes goblin↔spear-goblin flip-flop) + carry side-from-bar | ~2–4 days, no GPU |
| Archer levers (retrain): rebalance the 12× musketeer over-rep; add Duncan's-emulator real data | ~1 day + train |

The detector (v3) already out-recalls KataCR and reads side/spawn/level natively.
Remaining detector-quality issues (archers, flip-flop) are addressed here — the
flip-flop by the tracker (temporal), archers by rebalance + real data.

## Stage B — More real data (the quality lever, longer term)
Primary source is **SELF-PLAY** (Duncan's call): once the bot plays (Stage D), it
knows the card it played → auto-labels its own plays with certainty. TV Royale
auto-labeling (dual `cycle_tracker` → `localize_play` → `YoloDatasetWriter`,
already scaffolded) is the FALLBACK. Both depend on later stages, so near-term the
archer fix is rebalance, not a new pipeline. **Effort: rides on D.**

## Stage C — Post-2024 card coverage (parallel)
Cut sprites from the APK dumps for new cards → add to the generator pool →
retrain. **Effort ~2–4 days.**

## Stage D — Bot decision logic (Phase 5, the big one)
| Step | Work | Effort |
|---|---|---|
| D1 | Perception → clean board-state representation (much exists) | ~1 week |
| D2 | Rule-based policy (what/where to play) | weeks |
| D3 | Action execution (tap/place) — control layer exists, needs wiring | ~1 week |
| D4 | _(optional)_ learned policy (RL/offline; KataCR has jax reference) | open-ended |

**Rule-based playable bot: effort ~1–3 months · Calendar multi-month.** Learned
weights (D4): research-scale, open-ended.

## Headline numbers
- **Better detector than stock KataCR:** DONE (v3, real+synthetic).
- **Thin tracker (fixes flip-flop) + archer rebalance:** ~1 week.
- **Bot that actually plays (rule-based):** ~1–3 months.
- **Self-play self-labeling data engine:** comes online *with* the playing bot.
- **Learned decision weights:** open-ended beyond that — separate research.

## Biggest error bars
All of D (decision logic is the genuinely hard, open part). The detector and the
thin tracker are low-risk. Archers' residual live gap (KataCR-real vs Duncan's
emulator) is the one detector unknown, addressed by rebalance + self-play data.
