"""Train the HAND-CARD classifier -- a YOLOv8-cls model that reads which card is
in a hand slot, replacing the brittle template matching in the TV Royale hand
reader (and usable for live-play hand reading too).

Data comes from training/generate_hand_dataset.py (ImageNet-style train/val folders
of synthetic-from-icon crops + real in-hand crops). Same toolchain/GPU as the unit
detector (training/finetune.py); classification trains much faster than detection.

    python training/generate_hand_dataset.py --out training/datasets/hand_cls
    python training/finetune_hand_cls.py --data training/datasets/hand_cls

Resume:
    python training/finetune_hand_cls.py --model runs/classify/hand_cls/weights/last.pt --resume
"""

import argparse
import os

from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", help="Dataset dir (ImageNet-style train/val). Not needed with --resume.")
    p.add_argument("--model", default="yolov8s-cls.pt",
                   help="Start weights. Default yolov8s-cls.pt (ImageNet-pretrained classifier).")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=128, help="Slot crops are ~104-155px; 128 matches.")
    p.add_argument("--batch", type=int, default=-1, help="-1 = auto-fit VRAM.")
    p.add_argument("--device", default="0")
    p.add_argument("--name", default="hand_cls")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--patience", type=int, default=25, help="Early-stop patience.")
    args = p.parse_args()
    if not args.resume and not args.data:
        p.error("--data is required unless --resume is given")

    model = YOLO(args.model)
    if args.resume:
        model.train(resume=True)
        run_name = os.path.basename(os.path.dirname(os.path.dirname(args.model)))
    else:
        model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            name=args.name,
            patience=args.patience,
            cache=True,
        )
        run_name = args.name
    print("\nDone. Weights: runs/classify/%s/weights/best.pt" % run_name)
    print("Wrap it for the hand reader; preserve a snapshot to assets/models/ like the detector.")


if __name__ == "__main__":
    main()
