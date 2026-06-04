"""Verify whether vendor's FrameNum aligns with cv2 frame index.

Dumps frame N-1, N, N+1 from a video so we can compare against the
burn-in overlay to determine 0/1-indexing.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--frame", type=int, required=True, help="vendor FrameNum to probe")
    p.add_argument("--out-dir", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"failed to open: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    print(f"declared fps={fps}  total_frames={total}")

    for offset in (-1, 0, 1):
        idx = args.frame + offset
        if idx < 0 or idx >= total:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            print(f"failed to read frame {idx}")
            continue
        out = args.out_dir / f"frame_{idx}.png"
        cv2.imwrite(str(out), frame)
        print(f"wrote {out}")

    cap.release()


if __name__ == "__main__":
    main()
