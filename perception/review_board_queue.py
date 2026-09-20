"""Reviews the BOARD-UNIT crops that perception/board_collector banked while
watching TV Royale (assets/unlabeled_board_queue/), filing the ones you confirm
into assets/board_labeled/<card>/ as real, labelled examples. Those are then
turned into detector training data by training/prepare_board_data.py and folded
into the next YOLO retrain -- the loop that fixes the detector's confusable/gap
cards with real frames from YOUR emulator.

Deck knowledge makes labelling a 1-of-8 pick, not 1-of-150 (systems educate each
other): each crop shows the CANDIDATE cards from the unit's side's deck as a
numbered list, plus the collector's PROPOSED label:
  - a BOUND crop's proposal is the played card (ground truth) -> usually just Enter.
  - a STRUGGLE crop's proposal is the detector's (likely wrong) guess -> pick a number.
Controls per crop:
  Enter        accept the proposed label
  1..N         pick that candidate card
  <name>       type any card slug to override (fuzzy-matched; a typo is refused)
  x / d        discard (a bad crop -- half a unit, an effect, a spell splash)
  q            quit (unreviewed crops stay in the queue for next time)

RULE #2 (CLAUDE.md): nothing reaches the labelled library without this human step.
A card the detector CAN'T yet learn (a gap card with no detector class -- berserker,
ronin, ...) is still filed, but flagged; training/prepare_board_data.py holds those
aside until the taxonomy is extended, rather than silently dropping your label.

Usage:
    python -m perception.review_board_queue
"""

import json
import os
import shutil

import cv2

from game_knowledge.play_localizer import detected_classes, _card_ref

QUEUE_DIR = "assets/unlabeled_board_queue"
LABELED_DIR = "assets/board_labeled"
DISPLAY_MAX = 320   # crops are small; zoom up to this many px on the long side for a clean look


def _known_cards():
    return set(_card_ref().keys())


def _resolve(name, known):
    """Exact card slug, else close matches (difflib). Returns (slug|None, [suggestions])."""
    import difflib
    name = (name or "").strip().lower().replace(" ", "_").replace("-", "_")
    if name in known:
        return name, []
    return None, difflib.get_close_matches(name, known, n=5, cutoff=0.6)


def _show(img):
    h, w = img.shape[:2]
    s = DISPLAY_MAX / max(h, w)
    if s > 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_NEAREST)
    cv2.imshow("Review board crop (look here, answer in the terminal)", img)
    cv2.waitKey(1)


def _file(card, stem, img, meta):
    """Copy the crop + its (final-labelled) sidecar into assets/board_labeled/<card>/."""
    d = os.path.join(LABELED_DIR, card)
    os.makedirs(d, exist_ok=True)
    cv2.imwrite(os.path.join(d, stem + ".png"), img)
    meta = dict(meta, label=card)
    with open(os.path.join(d, stem + ".json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)


def main():
    if not os.path.isdir(QUEUE_DIR):
        print("No queue directory — nothing to review.")
        return
    stems = sorted(f[:-4] for f in os.listdir(QUEUE_DIR) if f.endswith(".png"))
    if not stems:
        print("Queue is empty — nothing to review.")
        return

    known = _known_cards()
    print(f"{len(stems)} board crop(s) queued.  Enter=accept  1..N=pick  name=override  x=discard  q=quit\n")
    filed = discarded = held = 0

    for i, stem in enumerate(stems, 1):
        png = os.path.join(QUEUE_DIR, stem + ".png")
        img = cv2.imread(png)
        if img is None:
            os.remove(png)
            continue
        meta = {}
        js = os.path.join(QUEUE_DIR, stem + ".json")
        if os.path.isfile(js):
            meta = json.load(open(js, encoding="utf-8"))

        proposed = meta.get("proposed")
        kind = meta.get("kind", "?")
        side = meta.get("side", "?")
        cands = meta.get("candidates") or []
        _show(img)

        opts = "  ".join(f"[{j}] {c}" for j, c in enumerate(cands, 1))
        prop_slug, _ = _resolve(proposed, known)
        prop_hint = f"Enter='{prop_slug}'" if prop_slug else "Enter=(no valid proposal, type one)"
        print(f"[{i}/{len(stems)}] {kind} crop, side={side}, proposed='{proposed}'.")
        if opts:
            print(f"    candidates: {opts}")

        while True:
            typed = input(f"    {prop_hint}, number, name, x=discard, q=quit: ").strip()
            low = typed.lower()
            if low == "q":
                cv2.destroyAllWindows()
                print(f"\nStopped early — {filed} filed ({held} gap-held), {discarded} discarded, "
                      f"{len(stems) - i + 1} left in the queue.")
                return
            if low in ("x", "d"):
                discarded += 1
                print("    Discarded.")
                chosen = None
                break
            if not typed:
                chosen = prop_slug
                if chosen is None:
                    print("    No valid proposal to accept — type a card name (or x).")
                    continue
                break
            if typed.isdigit() and 1 <= int(typed) <= len(cands):
                cand = cands[int(typed) - 1]
                slug, sugg = _resolve(cand, known)
                chosen = slug or cand   # candidates come from the deck; trust them even if not a ref key
                break
            slug, sugg = _resolve(typed, known)
            if slug is None:
                hint = ("  did you mean: " + ", ".join(sugg)) if sugg else " (no close match)"
                print(f"    '{typed}' isn't a known card.{hint}  Retype, or x.")
                continue
            chosen = slug
            break

        if chosen is not None:
            _file(chosen, stem, img, meta)
            trainable = bool(detected_classes(chosen))
            filed += 1
            if not trainable:
                held += 1
                print(f"    Filed as {chosen}  (gap card: no detector class yet — held for a "
                      f"taxonomy-extension retrain, not the current one).")
            else:
                print(f"    Filed as {chosen}.")
        # remove from the queue whether filed or discarded
        os.remove(png)
        if os.path.isfile(js):
            os.remove(js)

    cv2.destroyAllWindows()
    print(f"\nDone — {filed} filed ({held} gap-held), {discarded} discarded.\n"
          f"Build the training set:  python training/prepare_board_data.py")


if __name__ == "__main__":
    main()
