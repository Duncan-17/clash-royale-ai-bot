# Clash Royale AI — Pipeline & Tools Explained

A plain-English companion to the roadmap and philosophy docs. Those two explain *what* to build and *why*; this one explains *how the pieces that already exist actually work*, and defines the technical terms (pickle, YOLO, ONNX, etc.) as they come up, for someone who hasn't worked with this stuff before.

---

## 1. The pipeline, end to end

Every module in this project is part of one chain: turn a raw screenshot into facts a bot could act on.

```
LDPlayer (Android emulator running Clash Royale)
        │
        │  ADB (Android Debug Bridge — see §2)
        ▼
control/actions.py ──── sends taps/swipes back the other way
        │
        ▼
perception/capture.py  →  a screenshot, as a 900×1600 pixel image
        │
        ├──→ perception/card_detector.py   → "slot 1 = knight, 0.97 confidence"
        ├──→ perception/elixir_counter.py  → "elixir = 6.4"
        └──→ perception/unit_detector.py   → "knight at (x,y), ally"
                    │
                    ▼
             perception/unit_tracker.py    → same detection, but with a
                                              persistent ID across frames
        │
        ▼
   structured game state (numbers and labels, not pixels)
        │
        ▼
   [not built yet] decision-making — what the roadmap calls Phases 4+
```

Everything above the last arrow is **perception**: turning pixels into facts. Nothing in the current codebase decides what to *do* with those facts yet — that's deliberate, per the roadmap's phase order.

---

## 2. Glossary — the concepts, explained plainly

### ADB (Android Debug Bridge)
A tool that ships with Android that lets a program on your PC control an Android device (or, here, an Android *emulator* — LDPlayer pretending to be a phone) — take screenshots, send taps, install apps, etc. It's the standard, official way to automate Android from outside the device. `control/actions.py` and `perception/capture.py` both talk to LDPlayer over ADB.

### Template matching
The technique behind `card_detector.py` and `elixir_counter.py`. You keep a library of known reference images (e.g., 20 real photos of what a "Knight" card looks like in the hand). To identify a new crop, you slide it over each reference image and compute a similarity score (OpenCV's `cv2.matchTemplate` does this) — whichever reference scores highest, and clears a confidence threshold, is your answer. No machine learning involved — it's direct pixel comparison. Works great when you're matching against a small, known, visually-consistent set of things (122 cards, 11 elixir digits). Doesn't scale to "recognize any object in any position/lighting," which is where the next few concepts come in.

### Object detection / bounding box
A step up from template matching: instead of "is this crop card X or not," object detection answers "where are all the interesting things in this whole image, and what are they" — for each thing it finds, it draws a **bounding box** (the rectangle: left, top, right, bottom pixel coordinates) around it, names a class ("knight," "tesla," "musketeer"), and reports a **confidence score** (0–1, how sure the model is). This is what we need for unit detection: there could be any number of troops, anywhere on the board, and we don't know how many or where in advance — template matching doesn't work when you don't already know how many things you're looking for or where.

### YOLO ("You Only Look Once")
A specific, very popular family of object-detection neural networks. The name refers to its core trick: older detection approaches scanned an image in multiple passes (look for *this* class, then look for *that* class); YOLO looks at the whole image *once* and predicts all the boxes and classes simultaneously, which is why it's fast enough to run in real time on video. "YOLOv8," "YOLOv10," etc. are successive versions from different research groups/companies refining the same core idea. The unit-detection model we're integrating (`units_M_480x352.onnx`) is a **YOLOv10** model.

### Neural network / "trained" vs "fine-tuned" vs "from scratch"
A neural network is a large stack of numeric parameters (weights) that gets adjusted ("trained") by showing it many labeled examples until it gets good at a task. **Training from scratch** means starting those weights from random values. **Fine-tuning** means starting from a network someone *already* trained on a related task, and continuing training on your own smaller, more specific dataset — much faster than starting from zero, since the network already learned general visual concepts (edges, shapes, textures) and just needs to adapt them. The unit-detection model we're using was fine-tuned by build-a-bot's author on Clash Royale footage, starting from a generic YOLOv10 base.

### ONNX / ONNX Runtime
**ONNX** ("Open Neural Network Exchange") is a standard *file format* for trained neural networks — a way to save a model so it can be loaded and run by many different tools, not just the one that trained it. **ONNX Runtime** (`onnxruntime`, the Python package we installed) is Microsoft's engine for actually *running* (doing inference with) a model saved in that format — fast, works on CPU or GPU, doesn't need the original training framework installed. Important distinction: ONNX is an **inference-only** format — good for *using* a trained model, but you generally can't resume *training* it from the ONNX file the way you could from the original checkpoint. That's the crux of the "closing the 97-card gap" limitation discussed in Project_Status.

### Pickle, and why it matters here
`pickle` is Python's built-in way to save/load arbitrary Python objects to disk. It's relevant to mention because loading a `pickle` file (which is how raw PyTorch model checkpoints, `.pt` files, are usually saved) can **execute arbitrary code** if the file is malicious — Python has to run code to reconstruct the saved objects. This is a well-known security footgun with PyTorch checkpoints from untrusted sources. It's *not* a concern for the `.onnx` file we're using: ONNX stores a computation graph (a description of math operations), not a pickled Python object, so loading it doesn't execute arbitrary code the same way. This distinction is why the unit-detection model was treated as lower-risk than a random `.pt` download would be (see Project_Status's security discussion).

### Licenses (MIT vs AGPL, briefly)
**MIT** is a highly permissive open-source license — use it, modify it, redistribute it, basically no strings attached, just keep the copyright notice. **AGPL-3.0** is a "copyleft" license — if you distribute the software, or run it as a network service others use, you have to make your source available too. It does *not* restrict private personal use (running it yourself, not sharing it). The unit-detection model's embedded metadata claims AGPL-3.0 even though build-a-bot's own repo license is MIT — full discussion of what that means for this project is in Project_Status and the conversation that led to this doc.

---

## 3. What each perception module actually does

### `control/actions.py` — input
Wraps ADB taps and swipes. `EmulatorController.tap(x, y)`, `.swipe(x1,y1,x2,y2)`. No perception here — this is the "hands," not the "eyes."

### `perception/capture.py` — the eyes, raw feed
Returns the current screen as a color image (a numpy array, height×width×3), in the format OpenCV expects. Everything downstream starts from this. Originally pulled a fresh screenshot over ADB on every call (`adb screencap`) — reliable but slow (~550ms/call, since each call was a full round-trip: render, PNG-encode, transfer, decode). Rewritten (2026-08-15) around `adbnativeblitz`: a background thread keeps one continuous ADB video stream open (`adb shell screenrecord`, decoded live via PyAV) instead of repeated screenshot requests, so `screenshot()` just hands back whatever frame is already sitting in memory — ~0.5ms per call now, no round-trip at all. External behavior (what calling `screenshot()`/`save()` does) deliberately didn't change, so every other module needed zero updates. Documented gotcha already solved here: an early version had red and blue color channels swapped (a classic RGB-vs-BGR mixup between the PIL library and OpenCV) — fixed, see Project_Status for the story (the new adbnativeblitz path turned out to already return the right channel order, no conversion needed).

### `perception/card_detector.py` + friends — "what's in my hand"
**Technique: template matching.** Crops each of the 4 hand-card slots (fixed pixel positions, calibrated once — `config/hand_region.py`) and compares each crop against a library of real example images per card (`assets/hand_templates/<card_name>/`). `perception/auto_collect.py` builds that library automatically during real play — no manual labeling needed once a card is recognized once with high confidence.

### `perception/elixir_counter.py` — "how much elixir do I have"
**Technique: hybrid — template matching + raw pixel math.** The elixir number (0–10) is template-matched the same way cards are. The 10-segment elixir bar next to it is read by direct pixel-color analysis (no model, no templates needed) to get a continuous, fractional estimate (e.g. "6.4," meaning "at 6, 40% of the way to 7") — this is what tells you *when* the next point is coming, not just the current whole number.

### `perception/unit_detector.py` — "what troops/buildings are on the board"
**Technique: object detection via a pre-trained YOLOv10 model (ONNX format).** Unlike cards and elixir, this can't be simple template matching — units can be anywhere on the board, in any quantity, from either side, mid-animation, at varying scale/rotation over a busy background. This needs a real neural network. We're reusing a model from the `Pbatch/ClashRoyaleBuildABot` reference repo (`vendor/build-a-bot/`) rather than training our own from scratch, since a working, tested one already exists (MIT-licensed code, see §2 on the model's own AGPL metadata caveat).

What it does: takes the screenshot, crops out just the arena (cutting off the top/bottom UI — the model was trained on that same framing), resizes it to the model's expected input size (480×352 pixels), runs it through the network, and gets back a list of bounding boxes + class names ("knight," "musketeer," etc.) + confidence scores. Verified against real live gameplay (2026-08-15): 6/6 correct on the first real test, both box placement and class. Side (ally/enemy) is a SEPARATE step — see the `perception/side_reader.py` entry below; a second small model (`side.onnx`) originally did this but was removed entirely 2026-08-17 after proving too unreliable across every live test session.

Known limitation: this model recognizes 97 specific units/buildings, frozen as of its training date (2024-08-31) — any card added to the game after that won't be recognized by it (5 actual troop/building gaps identified: Berserker, Ronin, Spirit Empress, Suspicious Bush, Void — see Project_Status for the precise gap analysis, most of the raw 25-card diff is spells, which were never in scope for this detector). Closing it is harder than a quick fine-tune, since only the inference-ready ONNX export is available, not the original trainable weights. Unlike cards/elixir, plain template matching isn't a good fallback here either — board units appear at arbitrary position/scale/rotation over a changing background, the same reason this needed a real object-detection model in the first place. The realistic fix, deferred until it's an actual practical problem: use the existing model as an auto-labeler (pseudo-labeling) against our own footage to regenerate training coverage of the 97 known classes for free, hand-label just the missing card(s), and fine-tune a fresh public YOLOv10 checkpoint on the result.

Two supporting files feed into unit detection's filtering, both small and both worth knowing about: `config/red_zones.py` holds pixel regions where a detection should never legitimately land (the 6 towers, the river) — calibrated by clicking real screenshot coordinates and verified by drawing the zones back over a capture, not guessed. `perception/unit_metadata.py` holds per-unit AIR/GROUND transport data (needed so the river zone doesn't wrongly reject legitimate flying units) — reused from build-a-bot's own data, but cross-checked rather than trusted blindly (one real error found and fixed: their data listed Ice Wizard as AIR, which is wrong, it's a ground unit).

### `perception/unit_tracker.py` — "is this the same unit as last frame"
**Technique: algorithmic, nearest-neighbor matching between frames — no model involved.** `unit_detector.py` has no memory between calls; ask it twice in a row and it has no idea the second answer might be the same physical unit as the first. That's a problem in practice: a real unit's detection confidence naturally dips for a frame or two (mid-animation, slight rotation, another unit passing in front of it), which without tracking looks exactly like the unit vanished and a new one appeared — confirmed live, this reads as "units constantly being placed and dying" if you don't correct for it.

The fix: each frame, match new detections to already-known "tracks" by proximity (must be the same card name, same side, and within a pixel-distance threshold of where that track was last seen) — matched detections keep their existing ID, unmatched ones start a new one. A short "grace period" (a track survives a few consecutive missed frames before being dropped) bridges brief occlusion without losing identity. A second refinement (confirmation delay) requires a brand-new track to be matched twice before it's reported at all, which turned out to eliminate a large share of single-frame noise blips (real match data: 44% of all "tracks" in one match were never matched a second time — pure noise, now filtered out before ever surfacing).

---

## 4. Tools & dependencies — what each one is for

| Tool | What it is | Used for |
|---|---|---|
| **ADB** (`adb.exe`, bundled with LDPlayer) | Android's official device-control protocol | Screenshots, taps, swipes — the whole control/capture layer |
| **`pure-python-adb`** (`ppadb`) | A Python wrapper around talking to an ADB server | Cleaner than shelling out to `adb.exe` for every single command |
| **OpenCV** (`opencv-python`, imported as `cv2`) | A general-purpose computer vision library | Template matching (`cv2.matchTemplate`), image resizing/cropping, drawing debug overlays |
| **NumPy** (`numpy`) | Fast numeric array math for Python | Every image in this project is a NumPy array under the hood; all the pixel-level math (elixir bar fill %, etc.) is NumPy operations |
| **Pillow** (`PIL`) | Another image library | No longer actually used anywhere in the codebase (confirmed by search, 2026-08-15) — it used to decode screenshot bytes in the old capture.py, but the adbnativeblitz rewrite doesn't need it. Still listed in requirements.txt; flagged in Project_Status as a cleanup candidate rather than removed unilaterally |
| **PyYAML** | Reads `.yaml` config files | Loading `config/game_constants.yaml` (elixir timing) |
| **`requests`** | Makes HTTP calls | Fetching the card roster from the Clash Royale API, downloading card icon images |
| **`python-dotenv`** | Loads secrets from a `.env` file into environment variables | Keeping the Clash Royale API key out of source code |
| **`onnxruntime`** | Runs trained neural networks saved in the ONNX format | Runs the unit-detection and side (ally/enemy) models |
| **`adbnativeblitz`** | A small library that streams the emulator's screen as continuous video instead of repeated one-off screenshots | Backs `capture.py`'s fast screenshot path — see that module's writeup above |
| **`av`** (PyAV) | Python bindings for FFmpeg, the standard video encode/decode library | Pulled in automatically by `adbnativeblitz` — this is what actually decodes the video stream into individual frames |

Full current list always lives in `requirements.txt` — this table explains *why* each one is there.

---

## 5. Planned: training our own unit detector (not built yet)

Section 3 above described `unit_detector.py` reusing a third-party pre-trained model rather than training one from scratch. Real gameplay testing (see `Project_Status`) found that model's failure modes — units flickering in and out of existence for a couple of frames with nothing really there, whole classes appearing out of nowhere for cards that aren't even unlocked yet, and the separate ally/enemy model (`side.onnx`) flip-flopping on the same physical unit — aren't things a smarter program built on top of the model's output can fully fix. The model itself needs to get better, which means training our own, on our own footage. The blocker to that is the same one every object-detection project runs into: getting enough correctly-labeled examples (an image plus "here's the box around each unit, and what it is") is normally slow, manual work.

The plan gets around that using a feature of the actual Clash Royale app: **TV Royale**, a spectator mode for watching other players' matches, which — unlike playing your own match — shows **both** players' hands and elixir on screen at once. `game_knowledge/cycle_tracker.py` already knows how to watch a hand of cards and reliably figure out "a card just left this slot, so that card just got played" (see section 3's write-up... actually cycle_tracker isn't covered above; the short version: it watches for a hand slot's card to change to a new one, and double-checks the guess against how much elixir was spent, since every card has a known cost). Point that same logic at BOTH hands in a TV Royale broadcast, and you get a trustworthy answer to "who played what, and exactly when" for every single play in the match — without needing to look at the board at all.

That still leaves "where on the board did it land." For that, the plan reuses `unit_tracker.py`'s existing "is this a brand new unit, or one I've already seen" logic (already built for live play, described in section 3) to find whichever unit newly showed up right after a play was detected — but instead of trusting that system's own guess at *what* the unit is (the part that's unreliable), the guess gets overwritten with the trustworthy answer already known from the hand-watching step above. A new, much narrower piece (`perception/frame_diff.py` — comparing two images pixel-by-pixel to see what changed between them, a technique called frame differencing) only kicks in as a fallback, for the rarer case where the existing tracker misses a unit completely.

Put together with a script that keeps clicking "next replay" so it can run unattended (`data/tv_royale_watcher.py`), this turns into an automated pipeline that watches match after match and quietly builds up a labeled dataset (saved under `data/unit_training/`) with no one having to draw a single box by hand. Training an actual new model on that dataset is intentionally a separate, later step — this pipeline is just about generating the data it would train on.

Update, same day: this pipeline is now built (the data-generation pieces described above are real, tested code) but **paused before ever running live**, in favor of a cheaper fix tried first — see section 5b below. It isn't abandoned; if the cheaper fix doesn't close the gap, this is the fallback and doesn't need rebuilding.

---

## 5b. `perception/side_reader.py` — reading team color directly instead of guessing it (DONE, live-verified)

While designing the section 5 plan above, a deeper problem surfaced: `side.onnx` (the model that decided ally vs. enemy) never actually had a reliable signal to learn from. Clash Royale doesn't render an ally Giant and an enemy Giant differently — same sprite either way — so a model trying to guess team from an isolated crop was, at best, picking up on indirect and unstable clues (which way a unit happens to be facing at that instant, for example — which changes constantly and isn't really about team at all). That mismatch explained a lot of this project's hardest bugs: units getting misclassified right when they spawn, or flip-flopping mid-fight, were exactly the moments where whatever weak signal the model was leaning on broke down.

The actual answer turned out to be simpler than training a better model: Clash Royale already tells you the team, in the UI, directly. Every troop shows a small color-coded level number above it (red for the opponent's, blue for yours) and a matching color-coded health bar once it's taken any damage — with one asymmetry worth knowing: the opponent's troops always show their (red) level number, but your own troops show neither their level nor health bar until they've been hit at least once.

`perception/side_reader.py` reads that color directly, using the same "sample known pixels and check their color" technique `perception/elixir_counter.py` already validated for its own bar, not a learned model at all. It searches a region scaled to each unit's own bounding box (bar width and badge position both vary by unit/context, so a fixed offset doesn't work) for a red-family or blue-family blob of the right rough size/shape, then cross-checks that candidate against a small library of real digit-shape examples (same `cv2.matchTemplate` technique `card_detector.py`/`elixir_counter.py` already use) to reject color-matched-but-wrong-shape false positives — a boot, a unit's own hair, a piece of decorative UI — that color alone couldn't tell apart from a real badge. `side.onnx` was removed from `unit_detector.py` entirely once this replaced it. **Status: done, live-verified — a full match produced zero real side-reading mistakes** (see `Project_Status` for the detailed verification).

The one real gap color-reading can't cover on its own — a full-health, untouched troop of your own, showing no color cue at all — turned out to have a cleaner fix than the originally-planned own-play-timing fallback (below), addressed instead by the spawn-icon system in section 5c.

---

## 5c. `perception/spawn_icon_reader.py` — reading a card's side at the exact moment it's placed

A live review of `game_knowledge/opponent_play_tracker.py`'s output (32 real events, done with Duncan after side_reader.py itself tested clean) found the module's remaining mistakes all traced back to the same root cause: it has to *infer* "did the opponent just play a new card" from noticing a new detection box appear, and that inference breaks under occlusion, a unit dying instantly on spawn, or a single bad side-color read locking in permanently (see `perception/unit_tracker.py`'s known persistence-lock bug in `Project_Status`).

Duncan's own idea: Clash Royale shows a small circular gauge on a troop right as it's placed — a deploy-delay timer that fills clockwise, colored blue for an ally card and red for an enemy one, for BOTH sides regardless of damage state. Unlike a unit's ongoing health badge (which is damage-gated for allies), this shows for every fresh placement, which makes it a much more direct "was a card just played, and by whom" signal than inferring it from raw detection grouping. It also, as a side effect, closes section 5b's one remaining gap (an undamaged ally troop) for free, at the specific moment that matters most (when it's first placed) — no separate own-play-timing fallback needed.

The interesting technical wrinkle: unlike a badge's digit (a constant shape, with only a separate bar's fill % varying), the icon's entire visible content IS a continuously-changing pie-fill animation. A single reference screenshot at one fill level correlates poorly with the same real icon at a different fill level — template matching, which works well for digits, doesn't generalize here. So side comes directly from color instead (same underlying color families side_reader.py already validated), while shape/size filtering (plus keeping red and blue candidate search fully separate, so one real background element like a river doesn't corrupt the other's blob) does the job template matching does for badges. Integrated into `opponent_play_tracker.py` as the FIRST, highest-priority check ahead of its existing (much less direct) proximity-based heuristics.

---

## 6. Current status

This doc explains *how things work*, not *what's done right now* — that's `Project_Status`'s job, and it's kept current there. As of this doc's writing: capture/control (Phase 1) and game data (Phase 2) are done; card-in-hand detection, elixir counting, and unit/tower detection with tracking (Phase 3) are all done and validated live; reading team color directly (section 5b) and the spawn-icon system (section 5c) are both done and live-verified; Phase 4 (algorithmic game-knowledge — own-side card cycle tracking is done; opponent-side play detection has had many rounds of live-tested fixes and is close but not fully clean, see `Project_Status`'s "Next up") is in progress; the TV Royale auto-labeling pipeline (section 5) is built but paused, looking decreasingly likely to be needed now that 5b/5c have closed most of the gap it was meant to address.
