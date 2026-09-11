"""Fine-tune a standard Ultralytics YOLOv8 detector on a Clash Royale dataset,
on Duncan's RTX 5080. See training/README.md for the why (why STANDARD YOLOv8
and not KataCR's custom-head model, and where the data comes from).

Environment is already set up and verified (2026-09-07): torch 2.11.0+cu128,
ultralytics 8.4.123, CUDA on the RTX 5080 (16GB). A 1-epoch coco8 smoke train
ran clean on the GPU.

Usage:
    python training/finetune.py --data training/datasets/cr/data.yaml
    python training/finetune.py --data <data.yaml> --epochs 100 --imgsz 896 --batch -1

Resume / continue from a prior run's weights:
    python training/finetune.py --data <data.yaml> --model runs/detect/cr_finetune/weights/last.pt --resume
"""

import argparse
import os

from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", help="Path to the dataset data.yaml (YOLO format). Not needed with --resume.")
    p.add_argument("--model", default="yolov8l.pt",
                   help="Starting weights. Default yolov8l.pt (COCO-pretrained) -- KataCR's "
                        "detector*.pt are a custom head and can't seed a standard YOLO. Pass a "
                        "prior run's last.pt to continue.")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=896, help="896 matches KataCR's input size.")
    p.add_argument("--batch", type=int, default=-1, help="-1 = auto-fit VRAM (good on the 16GB 5080).")
    p.add_argument("--device", default="0", help="GPU index; '0' = the RTX 5080.")
    p.add_argument("--name", default="cr_finetune", help="Run name under runs/detect/.")
    p.add_argument("--resume", action="store_true", help="Resume an interrupted run from --model.")
    p.add_argument("--patience", type=int, default=30, help="Early-stop patience (epochs w/o improvement).")
    args = p.parse_args()
    if not args.resume and not args.data:
        p.error("--data is required unless --resume is given")

    model = YOLO(args.model)
    if args.resume:
        # Resume reads the run's saved config from the checkpoint (data, epochs,
        # name, ...) and continues where it left off -- passing those again would
        # conflict, so resume ONLY. --model must be that run's last.pt.
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
            # cache to RAM speeds up epochs; 5080 box has 31GB RAM
            cache=True,
            plots=True,
        )
        run_name = args.name
    print("\nDone. Weights: runs/detect/%s/weights/best.pt" % run_name)
    print("Evaluate/compare in the app with a wrapper around best.pt (see training/README.md).")


if __name__ == "__main__":
    main()
