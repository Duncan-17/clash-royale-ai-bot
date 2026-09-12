"""Inference wrapper for the hand-card classifier -- our YOLOv8-cls model
(assets/models/hand_cls/best.pt) that reads which card is in a hand slot.

Trained in training/ (generate_hand_dataset.py synthetic-from-icon + real crops;
finetune_hand_cls.py). Replaces the brittle template matching in the TV Royale
hand reader. classify_many() batches all a frame's slot crops into ONE predict
for speed.

Inputs are resized to IMGSZ square to match training (the generator saved
128x128 stretched crops), so the model sees the same framing at inference.
Optionally restrict predictions to the 16 deck cards of the current match
(allowed=), which is what makes reads reliable -- the argmax over 16 known cards.
"""

import cv2

DEFAULT_MODEL = "assets/models/hand_cls/best.pt"
IMGSZ = 128
SOFT_MARGIN = 0.25     # soft deck-restrict: an out-of-deck card is only taken over the
# best in-deck card if it beats it by at least this much probability. So a clear read
# of a card the deck was slightly misread on still gets through, but a weak misread
# snaps to the deck. Tunable.


def _soft_pick(p, allow, soft_margin=SOFT_MARGIN):
    """Index chosen from probs `p`: the overall best, unless it's outside `allow` and
    doesn't beat the best in-`allow` card by `soft_margin` (then snap to that in-set
    card). allow falsy -> plain argmax. See classify_many."""
    if not allow:
        return int(p.argmax())
    best = int(p.argmax())
    in_best = max(allow, key=lambda k: p[k])
    return best if (best in allow or p[best] - p[in_best] >= soft_margin) else in_best


class HandClassifier:
    def __init__(self, model_path=DEFAULT_MODEL, allowed=None):
        from ultralytics import YOLO
        import torch
        self.model = YOLO(model_path)
        self.names = self.model.names                 # {idx: card_slug}
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.set_allowed(allowed)

    def set_allowed(self, allowed):
        """Restrict predictions to these card slugs (e.g. the 16 deck cards), or
        None for the full 122-class output."""
        if allowed:
            allow = set(allowed)
            self.allowed_idx = [i for i, n in self.names.items() if n in allow]
            if not self.allowed_idx:
                self.allowed_idx = None
        else:
            self.allowed_idx = None

    def classify_many(self, crops, allowed_sets=None, soft_margin=SOFT_MARGIN):
        """[(card_slug, confidence)] for a list of BGR crops, in one batched predict.

        allowed_sets: optional per-crop iterable of card slugs to SOFT-restrict that
        crop to (e.g. the reading player's known deck). Soft = take the overall best
        card only if it's in the set OR beats the best in-set card by `soft_margin`;
        otherwise snap to the best in-set card. None for a crop = no restriction (falls
        back to the constructor's `allowed`, else unrestricted). Empty input -> []."""
        if not crops:
            return []
        imgs = [cv2.resize(c, (IMGSZ, IMGSZ)) if c.size else c for c in crops]
        res = self.model.predict(imgs, imgsz=IMGSZ, device=self.device, verbose=False)
        out = []
        for j, r in enumerate(res):
            p = r.probs.data.detach().cpu().numpy()
            allow = None
            if allowed_sets is not None and allowed_sets[j]:
                aset = set(allowed_sets[j])
                allow = [k for k, n in self.names.items() if n in aset]
            elif self.allowed_idx is not None:
                allow = self.allowed_idx
            i = _soft_pick(p, allow, soft_margin)
            out.append((self.names[i], float(p[i])))
        return out

    def classify(self, crop):
        """(card_slug, confidence) for one crop."""
        return self.classify_many([crop])[0]
