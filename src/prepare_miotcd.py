"""Prepare MIO-TCD localization subset in Ultralytics YOLO format.

MIO-TCD raw layout (after extracting MIO-TCD-Localization.tar):
    MIO-TCD-Localization/
        train/          # ~110k images (jpg)
        test/           # ~27k images (unlabeled)
        gt_train.csv    # filename,class,x1,y1,x2,y2  (one row per bbox)

This script:
    1. Reads gt_train.csv
    2. Does per-class stratified sampling (quotas in CLASS_QUOTAS)
    3. Copies selected images into data/miotcd_yolo/{train,val}/images/
    4. Writes YOLO-format labels to data/miotcd_yolo/{train,val}/labels/
    5. Emits summary to stdout

Usage:
    python src/prepare_miotcd.py \
        --raw-dir data/miotcd_raw/MIO-TCD-Localization \
        --out-dir data/miotcd_yolo \
        --val-ratio 0.1 \
        --seed 42
"""
from __future__ import annotations

import argparse
import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image

# MIO-TCD 11 classes. Ordering here becomes YOLO class IDs 0..10.
CLASSES = [
    "articulated_truck",
    "bicycle",
    "bus",
    "car",
    "motorcycle",
    "motorized_vehicle",
    "non-motorized_vehicle",
    "pedestrian",
    "pickup_truck",
    "single_unit_truck",
    "work_van",
]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}

# Stratified per-class image quotas (NOT bbox quotas).
# A single image may contain multiple classes; it is counted toward every class
# it contains and kept if ANY of its classes still has remaining quota.
# Values chosen to: keep all large-vehicle samples, down-sample "car" heavily.
CLASS_QUOTAS = {
    "car": 15000,
    "pickup_truck": 15000,
    "single_unit_truck": 20000,  # take all
    "articulated_truck": 15000,  # take all
    "bus": 15000,                # take all
    "work_van": 10000,
    "motorcycle": 5000,
    "bicycle": 5000,
    "pedestrian": 5000,
    "motorized_vehicle": 3000,
    "non-motorized_vehicle": 2000,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", type=Path, required=True,
                   help="Extracted MIO-TCD-Localization dir (contains train/ and gt_train.csv)")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Destination YOLO-format dataset dir")
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--link", action="store_true",
                   help="Symlink images instead of copying (saves disk, requires absolute paths)")
    return p.parse_args()


def load_annotations(csv_path: Path) -> dict[str, list[tuple[str, int, int, int, int]]]:
    """Return {image_id -> [(class_name, x1, y1, x2, y2), ...]}."""
    by_image: dict[str, list] = defaultdict(list)
    with csv_path.open() as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 6:
                continue
            img_id, cls, x1, y1, x2, y2 = row[0], row[1], int(row[2]), int(row[3]), int(row[4]), int(row[5])
            if cls not in CLASS_TO_ID:
                continue
            by_image[img_id].append((cls, x1, y1, x2, y2))
    return by_image


def stratified_select(by_image: dict, quotas: dict[str, int], seed: int) -> set[str]:
    """Greedy per-class fill. Shuffle images per class, add until quota hit.

    An image is 'selected' if it helps fill ANY class's quota. Because images
    are shared across classes, actual per-class counts often EXCEED quotas
    slightly — that's fine, it improves balance for rare classes.
    """
    rng = random.Random(seed)

    # Build per-class index: class -> list of image_ids
    class_to_images: dict[str, list[str]] = defaultdict(list)
    for img_id, boxes in by_image.items():
        seen_classes = {b[0] for b in boxes}
        for c in seen_classes:
            class_to_images[c].append(img_id)

    selected: set[str] = set()
    class_counts: dict[str, int] = defaultdict(int)

    # Process rare classes first so they get priority on shared images
    ordering = sorted(quotas.keys(), key=lambda c: len(class_to_images.get(c, [])))

    for cls in ordering:
        candidates = class_to_images.get(cls, [])
        rng.shuffle(candidates)
        quota = quotas[cls]
        for img_id in candidates:
            if class_counts[cls] >= quota:
                break
            if img_id not in selected:
                selected.add(img_id)
                # Bump counts for every class present in this image
                for c in {b[0] for b in by_image[img_id]}:
                    class_counts[c] += 1
            # If already selected via another class, it still counted toward cls above.
    print(f"[select] Total images selected: {len(selected)}")
    print("[select] Per-class image counts (>= quota for shared images):")
    for c in CLASSES:
        print(f"    {c:25s} {class_counts[c]:>7d}  (quota {quotas[c]})")
    return selected


def write_yolo_label(label_path: Path, boxes: list, img_w: int, img_h: int) -> None:
    lines = []
    for cls, x1, y1, x2, y2 in boxes:
        cid = CLASS_TO_ID[cls]
        # Clip and convert to normalized cxcywh
        x1 = max(0, min(x1, img_w - 1))
        x2 = max(0, min(x2, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        y2 = max(0, min(y2, img_h - 1))
        if x2 <= x1 or y2 <= y1:
            continue
        cx = (x1 + x2) / 2.0 / img_w
        cy = (y1 + y2) / 2.0 / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h
        lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""))


def materialize(selected: set[str], by_image: dict, raw_train: Path, out_dir: Path,
                val_ratio: float, seed: int, link: bool) -> None:
    rng = random.Random(seed + 1)
    ids = sorted(selected)
    rng.shuffle(ids)
    n_val = int(len(ids) * val_ratio)
    val_set = set(ids[:n_val])

    for split in ("train", "val"):
        (out_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    n_ok = n_bad = 0
    for i, img_id in enumerate(ids):
        split = "val" if img_id in val_set else "train"
        src_img = raw_train / f"{img_id}.jpg"
        if not src_img.exists():
            n_bad += 1
            continue
        dst_img = out_dir / split / "images" / f"{img_id}.jpg"
        dst_lbl = out_dir / split / "labels" / f"{img_id}.txt"
        if not dst_img.exists():
            if link:
                dst_img.symlink_to(src_img.resolve())
            else:
                shutil.copy2(src_img, dst_img)
        try:
            with Image.open(src_img) as im:
                w, h = im.size
        except Exception:
            n_bad += 1
            continue
        write_yolo_label(dst_lbl, by_image[img_id], w, h)
        n_ok += 1
        if (i + 1) % 5000 == 0:
            print(f"[write] {i+1}/{len(ids)}")
    print(f"[write] done. ok={n_ok} bad={n_bad}  train={len(ids)-n_val} val={n_val}")


def main():
    args = parse_args()
    csv_path = args.raw_dir / "gt_train.csv"
    train_dir = args.raw_dir / "train"
    assert csv_path.exists(), f"Missing {csv_path}"
    assert train_dir.exists(), f"Missing {train_dir}"

    print(f"[load] Reading {csv_path}")
    by_image = load_annotations(csv_path)
    print(f"[load] {len(by_image)} images have annotations")

    selected = stratified_select(by_image, CLASS_QUOTAS, args.seed)
    materialize(selected, by_image, train_dir, args.out_dir, args.val_ratio,
                args.seed, args.link)

    # Summary
    print(f"\n[done] Dataset ready at {args.out_dir}")
    print(f"       Point configs/miotcd.yaml at this directory.")


if __name__ == "__main__":
    main()
