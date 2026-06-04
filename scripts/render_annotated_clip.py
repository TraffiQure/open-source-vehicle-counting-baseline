"""Render an annotated MP4 of detector boxes + counting lines for visual debug.

For each frame in [start_frame, start_frame+num_frames) of --video, run YOLO+ByteTrack
and draw all detected boxes (with track id + MIO-TCD class label) plus the counting
lines from --lines. Output writes to --out as mp4.

Note: conf and tracker mirror src/run_baseline defaults so detections match the
production pipeline. This does NOT run the counter or stage-2 cascade -- it's
purely a "what does the detector see" visualization.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
from tqdm import tqdm

TRACKER = "bytetrack.yaml"
LINE_COLORS = {
    "West_01": (0, 255, 255),
    "West_02": (50, 200, 255),
    "East_01": (255, 255, 0),
    "East_02": (255, 200, 50),
}

# MIO-TCD detector classes (0..10 subset we route)
MIOTCD_NAMES = {
    0: "art_truck", 1: "bicycle", 2: "bus", 3: "car", 4: "motorcycle",
    5: "non_mot", 6: "pedestrian", 7: "pickup", 8: "single_unit_truck",
    9: "work_van", 10: "background",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--lines", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--num-frames", type=int, default=18000)
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--device", type=str, default="mps")
    p.add_argument("--crossings-csv", type=Path, default=None,
                   help="Optional ours PerVehicle CSV; flashes the matching line for FLASH_FRAMES on each crossing.")
    return p.parse_args()


FLASH_FRAMES = 20


def load_crossings(path: Path) -> dict:
    import csv as _csv
    out: dict[int, list[str]] = {}
    with path.open() as f:
        reader = _csv.reader(f)
        header_idx = None
        for i, r in enumerate(reader):
            if r and r[0].strip() == "CollectionTime":
                header_idx = i
                break
        if header_idx is None:
            return out
        for r in reader:
            if not r or not r[0].strip():
                continue
            try:
                direction = r[1].strip()
                lane = int(r[2])
                fnum = int(r[6])
            except (ValueError, IndexError):
                continue
            out.setdefault(fnum, []).append(f"{direction}_{lane:02d}")
    return out


def main() -> None:
    args = parse_args()
    from ultralytics import YOLO

    lines = json.loads(args.lines.read_text())["lines"]

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if args.start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args.out), fourcc, fps, (w, h))

    model = YOLO(str(args.model))
    crossings = load_crossings(args.crossings_csv) if args.crossings_csv else {}

    end_frame = min(total, args.start_frame + args.num_frames)
    n = end_frame - args.start_frame
    pbar = tqdm(total=n, unit="frame")
    fidx = args.start_frame
    t0 = time.perf_counter()
    try:
        while fidx < end_frame:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            results = model.track(
                frame, tracker=TRACKER, classes=list(range(11)),
                conf=args.conf, persist=True, verbose=False, device=args.device,
            )
            r = results[0] if results else None
            if r is not None and r.boxes is not None and len(r.boxes) > 0:
                xyxy = r.boxes.xyxy.cpu().numpy()
                cls = r.boxes.cls.cpu().numpy().astype(int)
                ids = r.boxes.id.cpu().numpy().astype(int) if r.boxes.id is not None else [None] * len(cls)
                confs = r.boxes.conf.cpu().numpy()
                for (x1, y1, x2, y2), c, tid, cf in zip(xyxy, cls, ids, confs):
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    label = f"{MIOTCD_NAMES.get(c, c)}"
                    if tid is not None:
                        label = f"#{tid} {label} {cf:.2f}"
                    cv2.putText(frame, label, (int(x1), max(int(y1) - 5, 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

            # any line that crossed within the last FLASH_FRAMES frames flashes red
            recent_flash: set[str] = set()
            for back in range(FLASH_FRAMES):
                for nm in crossings.get(fidx - back, []):
                    recent_flash.add(nm)

            for ln in lines:
                name = ln.get("name", "")
                flashing = name in recent_flash
                if flashing:
                    color = (0, 0, 255)
                    thickness = 8
                else:
                    color = LINE_COLORS.get(name, (200, 200, 200))
                    thickness = 2
                p1, p2 = ln["p1"], ln["p2"]
                cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, thickness)
                cv2.putText(frame, name, (int(p1[0]) + 4, int(p1[1]) - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                if flashing:
                    cv2.putText(frame, f"*** {name} CROSS ***",
                                (int(p1[0]) - 80, int(p1[1]) - 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.putText(frame, f"f={fidx} t={fidx/fps:.1f}s",
                        (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            writer.write(frame)
            fidx += 1
            pbar.update(1)
    finally:
        pbar.close()
        cap.release()
        writer.release()

    print(f"wrote {args.out}; {fidx - args.start_frame} frames; {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
