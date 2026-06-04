"""Overlay counting lines on a single video frame for visual verification."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


COLORS = {
    "West_01": (0, 255, 255),    # yellow
    "West_02": (0, 200, 0),      # green
    "East_01": (255, 0, 255),    # magenta
    "East_02": (255, 128, 0),    # orange
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--lines", type=Path, required=True)
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"open failed: {args.video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"read failed at frame {args.frame}")

    cfg = json.loads(args.lines.read_text())
    for line in cfg["lines"]:
        name = line["name"]
        color = COLORS.get(name, (255, 255, 255))
        p1 = tuple(line["p1"])
        p2 = tuple(line["p2"])
        cv2.line(frame, p1, p2, color, 2)
        label_pos = (p1[0] + 6, p1[1] - 4 if p1[1] > 12 else p1[1] + 14)
        cv2.putText(frame, name, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    cv2.imwrite(str(args.out), frame)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
