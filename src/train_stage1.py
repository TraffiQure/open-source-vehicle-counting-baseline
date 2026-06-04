"""Stage-1 fine-tune: YOLO11s on MIO-TCD subset.

Usage:
    python src/train_stage1.py
    python src/train_stage1.py --epochs 50 --imgsz 960 --batch 16

Notes:
    - Uses MPS on Apple Silicon; falls back to CPU if unavailable.
    - Output weights land in runs/detect/miotcd_yolo11s_v1/weights/best.pt.
    - Start from yolo11s.pt (COCO pretrained) — we fine-tune, not from scratch.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default=str(ROOT / "configs/miotcd.yaml"))
    p.add_argument("--weights", type=str, default=str(ROOT / "yolo11s.pt"))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--imgsz", type=int, default=640)  # MIO-TCD median is 720x480; 640 matches Ultralytics default
    p.add_argument("--batch", type=int, default=8)  # lowered from 16: MPS TAL shape-mismatch bug at higher batch
    p.add_argument("--device", type=str, default="mps")
    p.add_argument("--name", type=str, default="miotcd_yolo11s_v1")
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--workers", type=int, default=0)  # MPS DataLoader multiproc is unstable
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        patience=args.patience,
        workers=args.workers,
        project=str(ROOT / "runs/detect"),
        # Fine-tune friendly settings
        lr0=0.001,           # small LR since starting from COCO weights
        lrf=0.01,
        warmup_epochs=3,
        cos_lr=True,
        # Augmentation tuned for overhead surveillance:
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        hsv_h=0.015,
        hsv_s=0.4,           # reduced saturation jitter (many frames are near-grey)
        hsv_v=0.4,
        degrees=5.0,         # cameras don't rotate much
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        flipud=0.0,          # don't flip vertically, overhead view has a fixed up-direction
        # MPS stability
        amp=False,           # MPS AMP + TAL IoU has a known shape-broadcast bug; disable AMP
        # Save / log
        save=True,
        save_period=5,
        plots=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
