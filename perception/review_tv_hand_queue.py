"""Reviews the in-hand crops that perception.tv_royale_collector banked while
spectating (assets/unlabeled_tv_hand_queue/), filing the ones you confirm into
assets/hand_templates/ as real examples — this is how the library grows from
weak catalog seeds to TV-Royale-native templates.

Each crop shows the restricted matcher's best-of-16 GUESS (parsed from the
filename) as a starting suggestion, NOT a trusted label (RULE #2):
  - press Enter          -> accept the guess, file it under that card
  - type a card name     -> override, file under the name you type
  - type 'x' (or 'd')    -> discard (wrong crop, transition frame, etc.)

Nothing is filed into the permanent library without this step.

Usage:
    python -m perception.review_tv_hand_queue
"""

import os

import cv2

from perception.card_detector import add_example, known_card_slugs, resolve_card_name
from perception.tv_royale_collector import QUEUE_DIR

DISPLAY_SCALE = 2  # hand crops are ~106x132 -- a modest zoom for a clean read


def _parse(filename):
    """filename = '<guess>__<score>__<slot>__<ts>.png'. Returns (guess, score)."""
    parts = filename[:-4].split("__")
    guess = parts[0] if parts else "?"
    score = parts[1] if len(parts) > 1 else "?"
    return guess, score


def main():
    if not os.path.isdir(QUEUE_DIR):
        print("No queue directory found — nothing to review.")
        return
    files = sorted(f for f in os.listdir(QUEUE_DIR) if f.endswith(".png"))
    if not files:
        print("Queue is empty — nothing to review.")
        return

    print(f"{len(files)} crop(s) queued for review.\n")
    known = known_card_slugs()  # canonical names; typos get rejected, not filed
    filed, discarded = 0, 0

    for i, filename in enumerate(files, start=1):
        path = os.path.join(QUEUE_DIR, filename)
        img = cv2.imread(path)
        if img is None:
            os.remove(path)
            continue

        guess, score = _parse(filename)
        zoomed = cv2.resize(img, (img.shape[1] * DISPLAY_SCALE, img.shape[0] * DISPLAY_SCALE),
                            interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Review (look here, then answer in the terminal)", zoomed)
        cv2.waitKey(1)  # render without blocking

        # The guess from the filename can be stale (collected before a naming
        # fix, e.g. old 'minni_pekka') or an awkward icon spelling. Resolve it to
        # the EFFECTIVE name Enter will file: the guess if it's a valid card,
        # else its best close match. That way Enter always does something sensible
        # and the prompt shows exactly what -- never a dead end.
        guess_slug, suggestions = resolve_card_name(guess, known)
        effective = guess_slug or (suggestions[0] if suggestions else None)
        if guess_slug is None and suggestions:
            opts = "  ".join(f"[{j}] {s}" for j, s in enumerate(suggestions, 1))
            print(f"[{i}/{len(files)}] guess '{guess}' isn't a current card name — "
                  f"Enter will file the best match '{effective}'.  Or pick:  {opts}")

        # Loop until a valid decision. A name that isn't a known card is refused
        # (with numbered suggestions) rather than filed, so a typo can't create a
        # bogus folder; typing a suggestion NUMBER accepts it.
        while True:
            enter_hint = f"Enter=file '{effective}'" if effective else "type a name"
            typed = input(
                f"[{i}/{len(files)}] {enter_hint}, a name to override, a number, 'x'=discard: "
            ).strip()

            if typed.lower() in ("x", "d"):
                discarded += 1
                print("  Discarded.")
                break

            if not typed:  # Enter -> accept the effective name
                if effective is None:
                    print("  No valid guess to accept — type a card name (or 'x').")
                    continue
                add_example(effective, img)
                filed += 1
                print(f"  Filed as {effective}.")
                break

            if typed.isdigit() and 1 <= int(typed) <= len(suggestions):
                slug = suggestions[int(typed) - 1]
                add_example(slug, img)
                filed += 1
                print(f"  Filed as {slug}.")
                break

            slug, suggestions = resolve_card_name(typed, known)
            if slug is None:
                if suggestions:
                    opts = "  ".join(f"[{j}] {s}" for j, s in enumerate(suggestions, 1))
                    print(f"  '{typed}' isn't a known card. Pick a number or retype:  {opts}")
                else:
                    print(f"  '{typed}' isn't a known card and has no close match. Retype, or 'x' to discard.")
                continue
            add_example(slug, img)
            filed += 1
            print(f"  Filed as {slug}.")
            break

        os.remove(path)

    cv2.destroyAllWindows()
    print(f"\nDone — {filed} filed, {discarded} discarded.")


if __name__ == "__main__":
    main()
