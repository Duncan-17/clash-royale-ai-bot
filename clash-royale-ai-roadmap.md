# Clash Royale AI Bot — Master Roadmap & Resource Guide

Companion doc: `clash-royale-strategy-philosophy.md` covers *why* the bot values what it values (reward design, state features, module tagging). This doc covers *what to build, in what order, and what to reuse rather than build from scratch* — written to be handed directly to Claude Code as a working build plan.

---

## 1. Project shape

Three subsystems: **perception** (screenshot → structured facts), **control** (issuing taps/drags to an emulator), **decision-making** (the actual AI). Budget effort roughly 60% perception/plumbing, 25% decision-making, 15% training/iteration — perception quality caps everything downstream of it.

Three ways any given piece gets built (full definitions in the philosophy doc): **algorithmic** (hand-coded, deterministic — used where exact game data exists), **fine-tuned/hybrid** (start from an existing model or hand-designed base structure, let it refine further), **trained from scratch** (pure learning, used where there's no shortcut).

---

## 2. Unified build roadmap

Each phase lists the goal, what to build, and which resources from §3 to use instead of building blind.

### Phase 0 — Foundations (no code)
Goal: enough RL/PyTorch literacy to make informed decisions later.
- PyTorch "60 Minute Blitz" tutorial
- Hugging Face Deep RL course or OpenAI "Spinning Up in Deep RL"

### Phase 1 — Plumbing (control + capture, no AI)
Goal: reliable screenshot + tap/drag + start/end-game detection.
- Emulator: LDPlayer or BlueStacks
- ADB control: `pure-python-adb` (taps/drags)
- Fast capture: `scrcpy` or `adbnativeblitz` (avoid slow repeated `adb screencap` polling)
- Reference for plumbing patterns: **Pbatch/ClashRoyaleBuildABot**, **pyclashbot/py-clash-bot**

### Phase 2 — Runtime data + game constants
Goal: card roster/costs/combat stats and elixir timing available at startup, not hardcoded blind.
- Roster/elixir cost/rarity: official Clash Royale API (developer.clashroyale.com)
- Full combat stats (HP/damage/hit speed) — **not** reliably in the official API; pull from a community source (deckshop.pro, or a RoyaleAPI data mirror)
- Elixir regen timing (2.8s/1.4s/0.9s, phase boundaries) — fixed client rule, not API-exposed; keep as a config constant, spot-check periodically

### Phase 3 — Perception: cards, elixir, units
Goal: structured game state from a screenshot.
- Card-in-hand ID: OpenCV template matching (algorithmic — icon set is small and static, no model needed)
- Elixir count: pixel counting (algorithmic)
- Unit/tower/enemy detection: **originally planned** to fine-tune YOLOv8 from a Roboflow Universe Clash Royale model (§3); **actually shipped** by reusing Pbatch/ClashRoyaleBuildABot's pre-trained YOLOv10 (`units.onnx`/`side.onnx`) instead — see Project_Status for why, and its known limitations. Revised again 2026-08-16: those limitations (phantom short-lived detections, class hallucinations, unstable ally/enemy classification) turned out to be underlying-model problems no amount of downstream attribution logic could fully patch around, so the plan is now to train a **custom** detector on our own capture domain — see "Custom unit-detector training data pipeline" below, which supersedes the roboflow/inference serving plan (self-hosting a third-party model no longer applies once we're training our own).

### Phase 3b — Custom unit-detector training data pipeline (built 2026-08-16, PAUSED — Phase 3c closed the gap this was meant to address)
Goal: labeled training data for our own troop/building detector, generated largely unattended instead of hand-annotated. Full rationale and design in Project_Status ("Custom unit-detector training data pipeline"); summary:
- Watch Clash Royale's **TV Royale** spectator mode (shows both players' hands + elixir at once) with a `config/tv_royale_regions.py`-calibrated two-instance reuse of `game_knowledge/cycle_tracker.py`'s already-working own-side play detection — gives authoritative card+side+timestamp per play, no board-vision inference needed
- Localize each play on the board by reusing `perception/unit_tracker.py`'s existing new-track detection (overriding its class/side guess with the authoritative one above), with a narrow new `perception/frame_diff.py` fallback only for tracks the existing detector misses entirely
- `data/auto_labeler.py` writes standard YOLO-format examples to `data/unit_training/`, **class only** — side (ally/enemy) is deliberately NOT part of the taxonomy (corrected 2026-08-16, before any data was collected: Clash Royale doesn't skin-tint troops by side, so a per-crop class label like `giant_ally`/`giant_enemy` would just retrain side.onnx's same mistake at larger scale — see Phase 3c below for how side is actually being solved instead)
- `data/tv_royale_watcher.py` — the unattended "macro": navigates into TV Royale, runs the pipeline through each replay, advances to the next, repeats. Structurally complete but can't run yet — needs `config/tv_royale_regions.py` (a live calibration session, not written) and the actual TV Royale menu-navigation taps (UI flow not yet explored)
- Actual model training (installing `ultralytics`/`torch`, writing the fine-tune script) is a deliberate follow-up phase once real data volume exists, not part of this one
- **Status**: `perception/frame_diff.py` and `data/auto_labeler.py` are built and unit-tested (13 tests). Still paused before the remaining live-calibration work — Phase 3c (side_reader.py + the spawn-icon system) has now closed the ally/enemy gap this was meant to address cleanly enough (live-verified, zero real mistakes) that resuming this looks decreasingly likely to be needed at all. Not formally killed, just not currently worth the remaining calibration/menu-navigation effort — revisit only if a genuinely new gap shows up that Phase 3c's approach can't cover (e.g. the still-unfixed class-hallucination/goblin-detection-rate issues, which are base-model limitations no amount of Phase 3c-style algorithmic reading can fix).

### Phase 3c — Read team color directly instead of guessing it (DONE, live-verified 2026-08-17)
Goal: fix `side.onnx` specifically (ally/enemy classification), which turned out to be the dominant source of bugs across every live test this session — cheaper and more targeted than waiting on Phase 3b's full custom-model retrain. Full rationale in Project_Status ("Side (ally/enemy) determination — architecture correction").
- Clash Royale doesn't skin-tint troops by side, so a per-crop classifier (`side.onnx`, or a retrained model using a side-suffixed class taxonomy) has no reliable signal to learn — this is *why* side.onnx was unstable, not a training-data-volume problem
- The real signal: troops show a color-coded level number (always shown, red, for enemy troops; shown only once damaged, blue, for ally troops) and a matching color-coded health bar (also damage-gated for both sides), read directly and algorithmically — same technique `perception/elixir_counter.py` already validates for its own bar, not a learned model
- `perception/side_reader.py` — built, replaces `side.onnx` entirely in `perception/unit_detector.py`. Searches a region scaled to each detected unit's own bounding box (not a fixed pixel offset — bar width/badge position vary by unit). A second, independent digit-template crosscheck (`perception/label_level_badge.py`/`review_level_queue.py`, reusing the exact `card_detector.py`/`elixir_counter.py` matching technique) rejects color-matched-but-shape-wrong false positives (a boot, hair, a UI element) that pure color thresholds alone couldn't separate.
- Remaining gap (a fresh, undamaged ally troop shows no color signal at all) is now largely covered by Phase 4's spawn-icon system below, which reads side directly at the moment of placement regardless of damage state — the originally-planned `cycle_tracker.py`-based own-play-timing fallback was never built since the spawn icon turned out to be a cleaner, more direct solution to the same gap.
- **Status: DONE.** A full live match produced zero real side-reading mistakes (see Project_Status for the detailed verification, including 3 auto-captured "conflicts" that all traced to non-bugs on inspection). Not expected to need further dedicated work barring a genuinely new failure mode.

### Phase 4 — Algorithmic game-knowledge modules
Goal: the deterministic modules the philosophy doc calls out as "don't learn this, compute it."
- Card matchup table (1v1 combat calculator, exact stats from Phase 2 data) — not started
- Own-side cycle tracker (`game_knowledge/cycle_tracker.py`) — **DONE, live-verified**: reconstructs the hidden 4-card queue by elimination, cross-checked against elixir drops
- Opponent card-play detection (`game_knowledge/opponent_play_tracker.py`) — **built, multiple live-test rounds of fixes**, not yet fully clean. Attributes each newly-appeared enemy unit as played/spawned/uncertain/misclassified/reacquired (spawn-byproduct table in `game_knowledge/spawn_data.py`, ground-truth scoring in `game_knowledge/compare_ground_truth.py`). A 2026-08-17 review of 32 real events found several remaining bug categories (a `unit_tracker.py` side-persistence bug with no debounce, troop-tracking-gap double-counting, base-model class hallucinations) — see Project_Status "Next up" for current priority. A new **spawn/deploy-delay icon system** (`perception/spawn_icon_reader.py`) was added as the highest-priority signal in this module's attribution logic, reading the true moment-of-placement side directly from a color-coded UI icon rather than inferring it from raw detection grouping.
- Opponent deck/hand/elixir modeling (belief state over unknown slots, elixir inference) — not started, blocked on the above being solid first, per Duncan's own stated priority
- Cycle + elixir analysis (average/median/min/max cost, cycle-count-to-return-a-card) — not started

### Phase 5 — Scripted baseline bot
Goal: validate the full pipeline end-to-end before any learning enters the picture. Rule-based decisions on top of Phases 1–4's output. Isolates perception/control bugs from future learning bugs.

### Phase 6 — Hybrid modules (seeded + learned)
Goal: the modules the philosophy doc settled as "hand-designed base structure, learned refinement."
- Deck archetype classifier (seed with known archetypes — beatdown, cycle, siege, bait — let it form sub-archetypes as it sees more decks)
- Card banked-value curves (seed with base categories — tank/swarm/building/spell/support — learned mapping to actual value)

### Phase 7 — Replay-based imitation learning
Goal: a behavior-cloned policy trained entirely offline, in the background, no live emulator needed.
- Adapt the collection approach from **chrisrca/clash-royale-tv-replays** (§3) rather than build a TV Royale recorder from scratch
- Build the action-inference layer on top (frame-diffing: card disappears from hand + new unit appears = an action, at that elixir/position/time) — this doesn't exist publicly, it's the one piece you're building genuinely from scratch. **Partially pulled forward into Phase 3b** (2026-08-16): the core "detect a play, localize it on the board" mechanism turned out useful for generating unit-detector training data too, so a version of it is being built there first, live against TV Royale directly, rather than waiting for this phase.
- Train a PyTorch supervised classifier (state → action) on the resulting dataset

### Phase 8 — Reward function v1 → v2
Goal: move from a plain baseline reward to the philosophy doc's full design.
- v1 (ships as soon as Phase 4 exists): leak penalty, plain elixir differential, tower damage, win/loss
- v2 (needs Phase 6's archetype classifier online): win-path-weighted trade value — see philosophy doc §5.1

### Phase 9 — Reinforcement learning
Goal: fine-tune the Phase 7 imitation policy with RL rather than starting from random weights.
- Stable-Baselines3 (PPO or DQN) on a custom `gymnasium` wrapper
- Evaluate **samdickson22/clash-simulator** (§3) as the training environment before building a simulator from scratch — potentially thousands of times faster than training against the live client. Verify its current accuracy/maintenance state first; its own README hedges on some docs being outdated.

### Phase 10 — Emergent/learned layers
Goal: the pieces the philosophy doc explicitly says not to hand-code.
- Positional value not captured by health/damage (blocking a lock-on, saving a card)
- Emergent group-interaction outcomes on top of Phase 4's deterministic targeting rules
- Game-plan / win-path layer synthesizing archetype + tactical fundamentals — build this last, after Phases 1–9 are individually solid

### Phase 11 — Fine-tune against the real client, evaluate, iterate
Close any sim-to-real gap using the Phase 1 capture/control pipeline. Track win rate, trophy trend, reward/episode via TensorBoard or Weights & Biases. Keep a changelog between training runs.

---

## 3. Resource library

Everything found during research, organized for direct use. Cautions before cloning anything: (a) automating a live Clash Royale client violates Supercell's ToS — use a throwaway account; (b) these repos are for **study and architecture reference**, not guaranteed license-cleared for copy-paste reuse — check each repo's LICENSE file before importing code directly, several have no explicit license, which defaults to "all rights reserved" under GitHub's ToS even though the code is publicly visible.

### Reference bot projects — clone for architecture study

| Project | Clone | What it's for |
|---|---|---|
| Pbatch/ClashRoyaleBuildABot | `git clone https://github.com/Pbatch/ClashRoyaleBuildABot.git` | Perception + emulator-control plumbing reference (Phase 1). Actively maintained. |
| pyclashbot/py-clash-bot | `git clone https://github.com/pyclashbot/py-clash-bot.git` | Multi-emulator automation patterns (MEmu/BlueStacks/ADB). |
| samdickson22/clash-simulator | `git clone https://github.com/samdickson22/clash-simulator.git` | Full battle-sim engine, gymnasium-compatible, ~24,000x real-time — candidate RL training environment (Phase 9). Verify current state before committing. |
| MSU-AI/clash-royale-gym | `git clone https://github.com/MSU-AI/clash-royale-gym.git` | Gym env design reference (obs/action space shape). Early-stage, mock API — reference only. |
| Jaso1024/Real-Time-Strategy-RL-Clash-Royale | `git clone https://github.com/Jaso1024/Real-Time-Strategy-RL-Clash-Royale.git` | PPO + autoencoder state compression + factored action space reference. |
| 0xCyberstan/Royale-RL | `git clone https://github.com/0xCyberstan/Royale-RL-A-Reinforcement-Learning-Agent.git` | Full vision+RL pipeline reference (YOLOv9 + MobileNetV2 + EasyOCR + Decision Transformer), clean module separation. |
| houseworthe/claude-royale | `git clone https://github.com/houseworthe/claude-royale.git` | Different paradigm — LLM agent (Claude) plays directly, no training. Architecture reference for latency-hiding via parallel agents, not your main path. |

### Vision models/datasets

| Resource | How to get it | Notes |
|---|---|---|
| Roboflow Universe — Vision Bot's card detection model | https://universe.roboflow.com/vision-bot/clash-royale-card-detection-ylzsc | Actively updated. Free Roboflow account needed; download dataset+weights via the web UI, or programmatically: `pip install roboflow`, then `Roboflow(api_key=...).workspace("vision-bot").project("clash-royale-card-detection-ylzsc").version(N).download("yolov8")`. |
| Roboflow Universe — cleash's card detection model | https://universe.roboflow.com/cleash/clash-royale-card-detection-3ko0f | Same download method. |
| Roboflow Universe — full-frame Clash Royale detector | https://universe.roboflow.com/clashroyale/clash-royale-of3d3 | Broader detector (not just cards) — good starting checkpoint for unit/tower detection. |
| chrisrca/clash-royale-tv-replays | `pip install huggingface_hub`, then `huggingface-cli download chrisrca/clash-royale-tv-replays --repo-type dataset --local-dir ./tv-replays` | **1.88TB total — do not pull the whole thing.** Filter to specific arenas/replay IDs, or use the `datasets` library's streaming mode instead of a full local download. Frames + card-position labels only, no action labels (see Phase 7). |

### Self-hosted inference

| Resource | How to get it | Notes |
|---|---|---|
| roboflow/inference | `pip install inference inference-cli` then `inference server start` (auto-pulls Docker image) | Runs on port 9001; works on CPU, GPU, Jetson, even Raspberry Pi. Eliminates per-call API costs of hosted inference. |

### ADB / capture / control tooling

| Resource | How to get it | Notes |
|---|---|---|
| scrcpy | `git clone https://github.com/Genymobile/scrcpy.git` (or install via system package manager) | Fast screen streaming — fixes the slow-`adb-screencap` bottleneck for a live perception loop. |
| adbnativeblitz | `pip install adbnativeblitz` | Claims scrcpy-like speed without the scrcpy server dependency. |
| pure-python-adb (ppadb) | `pip install pure-python-adb` | Maintained Python ADB wrapper for taps/drags, avoids shelling out to raw `adb`. |
| usefuladbplus | `pip install usefuladbplus` | Alternative ADB automation wrapper. |

### Core ML/RL stack (pip installs)

```
pip install torch torchvision
pip install stable-baselines3
pip install gymnasium
pip install opencv-python
pip install ultralytics          # YOLOv8, for fine-tuning detection
pip install easyocr              # tower/HP number reading, if needed
pip install tensorboard          # or: pip install wandb
```

### Game/meta data sources

| Resource | URL | Notes |
|---|---|---|
| Official Clash Royale API | https://developer.clashroyale.com | Requires API key + IP whitelisting. Roster, elixir cost, rarity — not full combat stats. |
| RoyaleAPI | https://royaleapi.com | Deck/meta data, useful for archetype-classifier seeding (Phase 6) and opponent-deck priors (Phase 4). |
| deckshop.pro | https://www.deckshop.pro/card/list | Community-maintained full combat stats (HP/damage/hit speed) — fills the gap the official API leaves. |

---

## 4. Suggested starter repo layout

```
clash-royale-ai/
├── config/
│   └── game_constants.yaml       # elixir timing, refreshed card data cache
├── perception/
│   ├── capture.py                 # scrcpy/adb screen capture
│   ├── card_detector.py           # template matching
│   ├── elixir_counter.py          # pixel counting
│   └── unit_detector.py           # YOLOv8 wrapper, calls roboflow/inference
├── control/
│   └── actions.py                 # ADB taps/drags
├── game_knowledge/
│   ├── matchup_table.py           # Phase 4
│   ├── cycle_tracker.py           # Phase 4
│   └── archetype_classifier.py    # Phase 6
├── data/
│   ├── replay_collector.py        # Phase 7, adapted from chrisrca's approach
│   └── action_inference.py        # Phase 7, frame-diffing
├── models/
│   ├── imitation/                 # Phase 7 behavior-cloning model
│   └── rl_policy/                 # Phase 9 Stable-Baselines3 checkpoints
├── env/
│   └── clash_royale_env.py        # gymnasium wrapper (live client or clash-simulator)
├── reward/
│   └── reward_fn.py               # Phase 8, v1 then v2
└── vendor/                        # cloned reference repos from §3, study only
```

---

## 5. First setup commands

```bash
mkdir -p clash-royale-ai/vendor && cd clash-royale-ai

# reference repos (study, not direct import — check licenses)
git clone https://github.com/Pbatch/ClashRoyaleBuildABot.git vendor/build-a-bot
git clone https://github.com/samdickson22/clash-simulator.git vendor/clash-simulator
git clone https://github.com/0xCyberstan/Royale-RL-A-Reinforcement-Learning-Agent.git vendor/royale-rl

# core stack
pip install torch torchvision stable-baselines3 gymnasium opencv-python \
    ultralytics easyocr tensorboard inference inference-cli \
    pure-python-adb huggingface_hub
```
