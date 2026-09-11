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

    def classify_many(self, crops):
        """[(card_slug, confidence)] for a list of BGR crops, in one batched
        predict. Empty input -> []."""
        if not crops:
            return []
        imgs = [cv2.resize(c, (IMGSZ, IMGSZ)) if c.size else c for c in crops]
        res = self.model.predict(imgs, imgsz=IMGSZ, device=self.device, verbose=False)
        out = []
        for r in res:
            p = r.probs.data.detach().cpu().numpy()
            if self.allowed_idx is not None:
                i = max(self.allowed_idx, key=lambda k: p[k])
            else:
                i = int(p.argmax())
            out.append((self.names[i], float(p[i])))
        return out

    def classify(self, crop):
        """(card_slug, confidence) for one crop."""
        return self.classify_many([crop])[0]
