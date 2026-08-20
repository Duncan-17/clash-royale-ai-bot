# Working rules for this project

## RULE #1 — Never assert what's in an image without showing Duncan and getting confirmation (2026-08-17, repeatedly violated, take this seriously)

This is the single most important process rule in this file. A wrong assumption about what's in an image can silently corrupt calibration data, thresholds, or code with no way for Duncan to catch it after the fact — that's a real, serious cost, not a style preference. Re-stated and moved to the top 2026-08-17 after Claude cited an OLD image from memory/description ("this old reference image is your own confirmed ally Giant... its spawn icon is clearly red") instead of opening a FRESH copy with a new, unambiguous marker drawn on exactly the region being claimed — an old image's pre-existing annotations from a DIFFERENT earlier investigation (e.g. old "A"/"B" labels) are not a substitute for a fresh, purpose-built marker for the CURRENT claim.

Concretely, every time before drawing a conclusion from an image and acting on it (editing code, changing thresholds, confirming/denying a hypothesis):
1. Draw a FRESH marker (box/arrow/circle) directly on a crop, specifically highlighting the exact region the current claim is about — never reuse an old annotated image's labels for a new claim, even if it looks like the same picture.
2. Open that fresh image for Duncan (see "how to show an image" below) and explicitly ask him to confirm before treating the interpretation as settled.
3. Do not proceed to code/threshold changes based on the interpretation until he's confirmed it.

How to show an image:
- Duncan does not see tool-call outputs, only assistant text messages — reading an image with the Read tool does not display it to him.
- Open the actual file for him via Python launching a specific, always-present viewer directly, e.g. `subprocess.Popen(['mspaint.exe', path])` — not a generic OS "open with default app" call (`PowerShell Start-Process <path>` triggered an unwanted "choose an app" picker on his machine).
- When asking him to confirm something, name the specific region/box/color to look at rather than a general "does this look right?"

Store scratch/debug images in `temp_img/` (created 2026-08-16), not the project root. `test_screenshot.png` stays in the root — it's the canonical capture file referenced throughout the project's docs/quick-commands in `Project_Status`.

## Capturing screenshots

Never run `python -m perception.capture` (or anything else that overwrites `test_screenshot.png`) proactively. Duncan captures at his own chosen moment and tells you when it's ready — re-capturing on your own has previously overwritten his carefully-timed manual captures with a later, irrelevant frame (see `Project_Status`, `perception/side_reader.py` calibration history).

## RULE #2 — Never auto-write to a permanent asset library (assets/*_templates/) without Duncan's review

Added 2026-08-17 after a real incident: an auto-collection tool for `assets/spawn_icon_templates/` was built to auto-file detections directly with NO human check at all — reasoning that the side signal (color) was "confident enough" to skip review, unlike the queue-then-review pattern already established for cards/elixir/level-badges. Duncan caught a genuinely ambiguous real example that had already gone into the permanent library unreviewed, and was rightly bothered: he has always been the one who personally sorts/labels queued items before they become permanent training data, specifically so nothing gets guessed into a library he relies on without his oversight.

Concretely: every new detector's collection tool queues its detections to `assets/unlabeled_<thing>_queue/` for Duncan to review and label himself (see `perception/review_spawn_icon_queue.py` for the pattern) — it does NOT call the module's own `add_*_example()` function directly from the collection loop, no matter how confident the underlying signal seems. An existing collector auto-filing *without* a queue (e.g. cards/level-badges matching an established, already-validated template with high confidence) is fine to leave as-is — this rule is about never *introducing* a new unsupervised auto-file path, especially for a new/unvalidated detector.
