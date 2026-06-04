"""Stage-2 Layer 2 axle sub-classifier training.

Trains one axle-count classifier for either articulated_truck or
single_unit_truck crops using gold FHWA labels and a resolution-aware
augmentation/evaluation setup.

Usage:

    python -m src.train_stage2_axle \
        --vehicle-type articulated \
        --gold-csv outputs/axle_gold_articulated_v2.csv \
        --crop-dir data/miotcd_raw/train/articulated_truck \
        --out-dir runs/stage2_axle_articulated_v1 \
        --epochs 30 --batch 32
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
from torchvision.transforms import v2


VEHICLE_CONFIGS = {
    "articulated": {
        "labels": ["8", "9", "10", "multi_trailer"],
        "fhwa_to_idx": {8: 0, 9: 1, 10: 2, 11: 3, 12: 3, 13: 3},
        "modal_label": "9",
        "modal_acc": 0.78,
    },
    "single_unit": {
        "labels": ["5", "6", "7"],
        "fhwa_to_idx": {5: 0, 6: 1, 7: 2},
        "modal_label": "5",
        "modal_acc": 0.82,
    },
}
EXCLUDED_FHWA = {-1, -2}
TRAIN_FRAC = 0.8
VAL_FRAC = 0.2
IMG_SIZE = 224
VAL_DOWNSCALED_MIN_SIDE = 70
GATE_THRESHOLD = 0.55
WEIGHT_DECAY = 1e-4


class RandomResize:
    def __init__(self, min_side_low: int = 60, min_side_high: int = 120, p: float = 0.5):
        self.min_side_low = min_side_low
        self.min_side_high = min_side_high
        self.p = p

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return image
        w, h = image.size
        target_min_side = random.randint(self.min_side_low, self.min_side_high)
        scale = target_min_side / min(w, h)
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))
        return v2.Resize((new_h, new_w), antialias=True)(image)


class ResizeMinSide:
    def __init__(self, min_side: int):
        self.min_side = min_side

    def __call__(self, image: Image.Image) -> Image.Image:
        w, h = image.size
        scale = self.min_side / min(w, h)
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))
        return v2.Resize((new_h, new_w), antialias=True)(image)


class AxleDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, int]], transform):
        self.samples = samples
        self.targets = [label for _, label in samples]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        with Image.open(path) as image:
            image = image.convert("RGB")
        image = self.transform(image)
        return image, label


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vehicle-type", required=True, choices=["articulated", "single_unit"])
    p.add_argument("--gold-csv", required=True, type=Path)
    p.add_argument("--crop-dir", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--balance-mode", choices=["sampler", "loss", "both", "none"], default="sampler")
    p.add_argument("--freeze-backbone", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--dropout", type=float, default=0.3)
    return p.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_transforms():
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    train_tf = T.Compose([
        RandomResize(p=0.5),
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToTensor(),
        normalize,
    ])
    val_clean_tf = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        normalize,
    ])
    val_downscaled_tf = T.Compose([
        ResizeMinSide(VAL_DOWNSCALED_MIN_SIDE),
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        normalize,
    ])
    return train_tf, val_clean_tf, val_downscaled_tf


def load_samples(gold_csv: Path, crop_dir: Path, fhwa_to_idx: dict[int, int]):
    samples: list[tuple[Path, int]] = []
    skipped_excluded = 0
    skipped_missing = 0
    with gold_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fhwa_class = int(row["fhwa_class"])
            if fhwa_class in EXCLUDED_FHWA:
                skipped_excluded += 1
                continue
            path = crop_dir / row["filename"]
            if not path.exists():
                skipped_missing += 1
                continue
            samples.append((path, fhwa_to_idx[fhwa_class]))
    return samples, skipped_excluded, skipped_missing


def stratified_split(samples: list[tuple[Path, int]], seed: int):
    by_class: dict[int, list[tuple[Path, int]]] = defaultdict(list)
    for sample in samples:
        by_class[sample[1]].append(sample)

    rng = random.Random(seed)
    train_samples: list[tuple[Path, int]] = []
    val_samples: list[tuple[Path, int]] = []
    for label in sorted(by_class):
        class_samples = by_class[label][:]
        rng.shuffle(class_samples)
        n_total = len(class_samples)
        n_val = int(round(n_total * VAL_FRAC))
        if n_total > 1:
            n_val = min(max(n_val, 1), n_total - 1)
        else:
            n_val = 0
        val_samples.extend(class_samples[:n_val])
        train_samples.extend(class_samples[n_val:])
    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    return train_samples, val_samples


def build_loaders(train_samples, val_samples, batch_size: int, seed: int, balance_mode: str):
    train_tf, val_clean_tf, val_downscaled_tf = make_transforms()
    train_ds = AxleDataset(train_samples, transform=train_tf)
    val_clean_ds = AxleDataset(val_samples, transform=val_clean_tf)
    val_downscaled_ds = AxleDataset(val_samples, transform=val_downscaled_tf)

    train_counts = Counter(train_ds.targets)
    sampler = None
    shuffle = False
    if balance_mode in {"sampler", "both"}:
        sample_weights = [1.0 / (train_counts[label] ** 0.5) for label in train_ds.targets]
        sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(train_ds),
            replacement=True,
            generator=torch.Generator().manual_seed(seed),
        )
    else:
        shuffle = True

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              shuffle=shuffle, num_workers=0, pin_memory=False)
    val_clean_loader = DataLoader(val_clean_ds, batch_size=batch_size, shuffle=False,
                                  num_workers=0, pin_memory=False)
    val_downscaled_loader = DataLoader(val_downscaled_ds, batch_size=batch_size, shuffle=False,
                                       num_workers=0, pin_memory=False)
    return train_ds, val_clean_ds, val_downscaled_ds, train_loader, val_clean_loader, val_downscaled_loader


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += x.size(0)
    return correct / total


def get_git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            cwd=Path(__file__).resolve().parent.parent,
        ).strip()
    except Exception:
        return None


def save_checkpoint(path: Path, model: nn.Module, args, classes: list[str], epoch: int,
                    train_acc: float, val_clean_acc: float, val_downscaled_acc: float):
    torch.save({
        "model": model.state_dict(),
        "classes": classes,
        "vehicle_type": args.vehicle_type,
        "epoch": epoch,
        "train_acc": train_acc,
        "val_clean_acc": val_clean_acc,
        "val_downscaled_acc": val_downscaled_acc,
        "args": vars(args),
    }, path)


def main():
    args = parse_args()
    seed_everything(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    config = VEHICLE_CONFIGS[args.vehicle_type]
    classes = config["labels"]

    samples, skipped_excluded, skipped_missing = load_samples(
        args.gold_csv,
        args.crop_dir,
        config["fhwa_to_idx"],
    )
    if not samples:
        raise ValueError("No usable training samples found")

    train_samples, val_samples = stratified_split(samples, args.seed)
    train_counts = Counter(label for _, label in train_samples)
    val_counts = Counter(label for _, label in val_samples)

    print(f"Vehicle type: {args.vehicle_type}")
    print(f"Usable samples: {len(samples):,}  -> train {len(train_samples):,} / val {len(val_samples):,}")
    print(f"Skipped excluded labels (-1/-2): {skipped_excluded:,}")
    print(f"Skipped missing files: {skipped_missing:,}")
    print("Train class distribution:")
    for idx, name in enumerate(classes):
        count = train_counts.get(idx, 0)
        print(f"  {idx} {name:<14s} {count:>5,}")
        if count < 10:
            print(f"WARNING: train class '{name}' has only {count} samples after split")
    print("Val class distribution:")
    for idx, name in enumerate(classes):
        print(f"  {idx} {name:<14s} {val_counts.get(idx, 0):>5,}")

    train_ds, _, _, train_loader, val_clean_loader, val_downscaled_loader = build_loaders(
        train_samples,
        val_samples,
        args.batch,
        args.seed,
        args.balance_mode,
    )

    device = get_device()
    print(f"Device: {device}")

    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=args.dropout, inplace=True),
        nn.Linear(in_features, len(classes)),
    )
    if args.freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False
    model.to(device)

    ce_weights = None
    if args.balance_mode in {"loss", "both"}:
        ce_weights = torch.tensor(
            [1.0 / (train_counts[idx] ** 0.5) for idx in range(len(classes))],
            dtype=torch.float32,
            device=device,
        )
    criterion = nn.CrossEntropyLoss(weight=ce_weights)
    optim_params = model.classifier.parameters() if args.freeze_backbone else model.parameters()
    optim = torch.optim.AdamW(optim_params, lr=args.lr, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim,
        T_max=args.epochs,
        eta_min=1e-6,
    )

    classes_path = args.out_dir / "classes.json"
    classes_path.write_text(json.dumps({idx: name for idx, name in enumerate(classes)}, indent=2) + "\n")

    config_path = args.out_dir / "config.json"
    config_path.write_text(json.dumps({
        **vars(args),
        "git_sha": get_git_sha(),
        "device": str(device),
        "train_frac": TRAIN_FRAC,
        "val_frac": VAL_FRAC,
    }, indent=2, default=str) + "\n")

    log_path = args.out_dir / "train_log.csv"
    log_f = log_path.open("w", newline="")
    log_w = csv.writer(log_f, lineterminator="\n")
    log_w.writerow(["epoch", "train_loss", "train_acc", "val_clean_acc", "val_downscaled_acc", "lr"])

    best_downscaled_acc = -1.0
    best_clean_acc = 0.0
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        model.train()
        if args.freeze_backbone:
            model.features.eval()
        tr_loss = 0.0
        tr_correct = 0
        tr_total = 0
        for i, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            optim.zero_grad(set_to_none=True)
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optim.step()
            tr_loss += loss.item() * x.size(0)
            tr_correct += (out.argmax(1) == y).sum().item()
            tr_total += x.size(0)
            if i % 100 == 0:
                print(f"  ep {epoch} it {i}/{len(train_loader)} "
                      f"loss={loss.item():.3f} running_acc={tr_correct / max(tr_total, 1):.3f}",
                      flush=True)

        tr_loss /= tr_total
        tr_acc = tr_correct / tr_total
        val_clean_acc = evaluate(model, val_clean_loader, device)
        val_downscaled_acc = evaluate(model, val_downscaled_loader, device)
        sched.step()
        lr_now = optim.param_groups[0]["lr"]
        epoch_sec = time.perf_counter() - t0

        print(f"[ep {epoch}/{args.epochs}] "
              f"train loss={tr_loss:.3f} acc={tr_acc:.3f} | "
              f"val clean acc={val_clean_acc:.3f} | "
              f"val downscaled acc={val_downscaled_acc:.3f} | "
              f"lr={lr_now:.2e} | {epoch_sec:.0f}s")

        log_w.writerow([epoch, f"{tr_loss:.6f}", f"{tr_acc:.6f}",
                        f"{val_clean_acc:.6f}", f"{val_downscaled_acc:.6f}",
                        f"{lr_now:.8f}"])
        log_f.flush()

        save_checkpoint(args.out_dir / "last.pt", model, args, classes, epoch,
                        tr_acc, val_clean_acc, val_downscaled_acc)
        if val_downscaled_acc > best_downscaled_acc:
            best_downscaled_acc = val_downscaled_acc
            best_clean_acc = val_clean_acc
            best_epoch = epoch
            save_checkpoint(args.out_dir / "best.pt", model, args, classes, epoch,
                            tr_acc, val_clean_acc, val_downscaled_acc)
            print(f"  * new best downscaled-val acc={val_downscaled_acc:.3f}, saved best.pt")

    log_f.close()

    modal_text = (
        f"articulated->Class {VEHICLE_CONFIGS['articulated']['modal_label']} = "
        f"~{VEHICLE_CONFIGS['articulated']['modal_acc'] * 100:.0f}%, "
        f"single_unit->Class {VEHICLE_CONFIGS['single_unit']['modal_label']} = "
        f"~{VEHICLE_CONFIGS['single_unit']['modal_acc'] * 100:.0f}%"
    )
    decision = "PASS (ship)" if best_downscaled_acc >= GATE_THRESHOLD else "FAIL (fall back to modal class)"

    print(f"Done. Best epoch by downscaled val: {best_epoch}")
    print(f"Best clean val accuracy at best-downscaled epoch: {best_clean_acc * 100:.2f}%")
    print(f"Best downscaled val accuracy: {best_downscaled_acc * 100:.2f}%")
    print("=== Ship gate ===")
    print(f"Best downscaled-val accuracy: {best_downscaled_acc * 100:.2f}%")
    print(f"Gate threshold: {GATE_THRESHOLD * 100:.2f}%")
    print(f"Decision: {decision}")
    print(f"Modal fallback accuracy (would-be): {modal_text}")


if __name__ == "__main__":
    main()
