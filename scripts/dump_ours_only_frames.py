"""Dump strips at ours timestamps where vendor has no same-direction
crossing within +/- match-window. Mirror of dump_mismatch_frames.py
but with ours/vendor roles swapped. Output is compatible with
scripts/review_mismatch_gui.py."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2

from dump_mismatch_frames import (
    CrossingRecord,
    build_direction_index,
    direction_key,
    ensure_output_dir,
    format_csv_timestamp,
    format_delta,
    load_ours,
    load_vendor,
    nearest_same_direction,
    render_strip,
    resolve_video_start_dt,
    timestamp_to_frame_idx,
    warn,
    write_index,
)


@dataclass(frozen=True, slots=True)
class OursOnlyCase:
    ours: CrossingRecord
    vendor_nearest: CrossingRecord | None
    delta_s: float | None
    frame_idx: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ours", type=Path, required=True)
    p.add_argument("--vendor", type=Path, required=True)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--direction", type=str, required=True, help="East / West")
    p.add_argument("--lane", type=str, required=True, help="ours lane number, e.g. 1")
    p.add_argument("--window-s", type=float, default=5.0)
    p.add_argument("--max-cases", type=int, default=None)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--strip-half-window-frames", type=int, default=6)
    p.add_argument("--strip-stride", type=int, default=1)
    p.add_argument("--video-start-time", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_output_dir(args.out_dir)

    ours_rows = load_ours(args.ours, args.video)
    vendor_rows = load_vendor(args.vendor)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"open failed: {args.video}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        if fps <= 0 or frame_count <= 0:
            raise SystemExit("bad video metadata")

        window_start = min(r.timestamp for r in ours_rows)
        window_end = max(r.timestamp for r in ours_rows)
        video_t0 = resolve_video_start_dt(args.video, window_start, args.video_start_time)

        vendor_index = build_direction_index(vendor_rows)
        dir_key = direction_key(args.direction)

        cases: list[OursOnlyCase] = []
        for o in ours_rows:
            if direction_key(o.direction) != dir_key:
                continue
            if o.lane != args.lane:
                continue
            if o.timestamp < window_start or o.timestamp > window_end:
                continue
            v_nearest, delta = nearest_same_direction(vendor_index, o)
            if delta is not None and abs(delta) <= args.window_s:
                continue  # vendor has match -> not ours-only
            # ours pipeline writes its own cv2 frame index into FrameNum,
            # so use it directly. Don't go through ours timestamp
            # (which is filename-anchored and ~18s off real video t0).
            try:
                frame_idx = int(o.frame_num.strip())
            except (ValueError, AttributeError):
                warn(f"skipping ours {format_csv_timestamp(o.timestamp)}: bad FrameNum {o.frame_num!r}")
                continue
            cases.append(OursOnlyCase(o, v_nearest, delta, frame_idx))
            if args.max_cases is not None and len(cases) >= args.max_cases:
                break

        index_rows = []
        saved = skipped = 0
        for case in cases:
            if case.frame_idx < 0 or case.frame_idx >= frame_count:
                skipped += 1
                warn(f"skipping {format_csv_timestamp(case.ours.timestamp)} frame {case.frame_idx} out of [0, {frame_count - 1}]")
                continue
            strip = render_strip(
                capture=cap,
                center_frame_idx=case.frame_idx,
                fps=fps,
                frame_count=frame_count,
                half_window_frames=args.strip_half_window_frames,
                stride=args.strip_stride,
            )
            fname = (
                f"{case.ours.timestamp.strftime('%H%M%S')}_o{case.ours.vehicle_class}_"
                f"{case.ours.video_file_num}_{case.ours.frame_num}.png"
            )
            strip_path = args.out_dir / fname
            strip.save(strip_path)
            # Note: GUI reads ours_nearest_west_*; we put OURS row into
            # the vendor_* fields and VENDOR-nearest into ours_nearest_west_*
            # so the same GUI works (label is direction-agnostic per spec).
            index_rows.append({
                "case_idx": str(saved + 1),
                "vendor_time": format_csv_timestamp(case.ours.timestamp),
                "vendor_lane": case.ours.lane + " (ours)",
                "vendor_class": case.ours.vehicle_class + " (ours)",
                "ours_nearest_west_time": (
                    format_csv_timestamp(case.vendor_nearest.timestamp)
                    if case.vendor_nearest is not None else ""
                ),
                "ours_nearest_west_delta_s": format_delta(case.delta_s),
                "ours_nearest_west_lane": (
                    case.vendor_nearest.lane + " cls=" + case.vendor_nearest.vehicle_class + " (vendor)"
                    if case.vendor_nearest is not None else ""
                ),
                "strip_path": str(strip_path),
            })
            saved += 1
        write_index(args.out_dir / "index.csv", index_rows)
    finally:
        cap.release()

    print(f"ours {args.direction} lane={args.lane} with no vendor match within +/-{args.window_s}s: {len(cases)}")
    print(f"saved {saved} strips to {args.out_dir}")
    if skipped:
        print(f"skipped {skipped} out-of-range")


if __name__ == "__main__":
    main()
