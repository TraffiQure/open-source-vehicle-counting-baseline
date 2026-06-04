"""Overlay counting lines and crossing events on video for manual validation."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2

LINE_COLORS = {
    "West_01": (0, 255, 255),
    "West_02": (50, 200, 255),
    "East_01": (255, 255, 0),
    "East_02": (255, 200, 50),
}

FLASH_FRAMES = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--lines", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=1200)
    return parser.parse_args()


def load_crossings(csv_path: Path) -> dict[int, list[str]]:
    text = csv_path.read_text(encoding="utf-8")
    rows = text.splitlines()
    header_idx = next(i for i, r in enumerate(rows) if r.startswith("CollectionTime"))
    crossings: dict[int, list[str]] = defaultdict(list)
    for row in rows[header_idx + 1 :]:
        parts = [p.strip() for p in row.split(",")]
        if len(parts) < 7 or not parts[0]:
            continue
        try:
            direction = parts[1]
            lane = int(parts[2])
            frame_num = int(parts[6])
        except (ValueError, IndexError):
            continue
        crossings[frame_num].append(f"{direction}_{lane:02d}")
    return crossings


def main() -> None:
    args = parse_args()
    lines_json = json.loads(args.lines.read_text(encoding="utf-8"))
    lines = lines_json["lines"]
    crossings = load_crossings(args.csv)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise ValueError(f"Could not open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if args.start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args.out), fourcc, fps, (w, h))

    totals: Counter[str] = Counter()
    flash: dict[str, int] = {}
    window_hits: Counter[str] = Counter()

    for i in range(args.num_frames):
        frame_idx = args.start_frame + i
        ok, frame = cap.read()
        if not ok:
            break

        for line_name in crossings.get(frame_idx, []):
            totals[line_name] += 1
            window_hits[line_name] += 1
            flash[line_name] = FLASH_FRAMES

        for ln in lines:
            name = ln["name"]
            p1 = (int(ln["p1"][0]), int(ln["p1"][1]))
            p2 = (int(ln["p2"][0]), int(ln["p2"][1]))
            flashing = flash.get(name, 0) > 0
            color = (0, 0, 255) if flashing else LINE_COLORS.get(name, (0, 255, 0))
            thickness = 5 if flashing else 2
            cv2.line(frame, p1, p2, color, thickness)
            label_pos = (p1[0] + 6, min(p1[1], p2[1]) - 4)
            cv2.putText(
                frame, name, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1
            )
            if flashing:
                mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
                cv2.circle(frame, mid, 14, (0, 0, 255), 2)

        for name in list(flash):
            flash[name] -= 1
            if flash[name] <= 0:
                del flash[name]

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (170, 110), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

        cv2.putText(
            frame, f"Frame {frame_idx}", (6, 16),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
        )
        y = 34
        for name in ["West_01", "West_02", "East_01", "East_02"]:
            cv2.putText(
                frame, f"{name}: {totals[name]}", (6, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, LINE_COLORS[name], 1,
            )
            y += 18

        writer.write(frame)

    cap.release()
    writer.release()

    print(f"Wrote {args.out}")
    print(f"Window [{args.start_frame}-{args.start_frame + args.num_frames}]: "
          f"{dict(window_hits)}")


if __name__ == "__main__":
    main()
