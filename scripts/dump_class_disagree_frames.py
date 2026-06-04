"""Dump strips at vendor timestamps where vendor-class is X but
ours-nearest-same-direction-within-window is class Y. Output is
compatible with scripts/review_mismatch_gui.py."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2

from dump_mismatch_frames import (
    CrossingRecord,
    build_direction_index,
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
class DisagreeCase:
    vendor: CrossingRecord
    ours: CrossingRecord
    delta_s: float
    frame_idx: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ours", type=Path, required=True)
    p.add_argument("--vendor", type=Path, required=True)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--vendor-class", type=str, required=True)
    p.add_argument("--ours-class", type=str, required=True)
    p.add_argument("--match-window-s", type=float, default=5.0)
    p.add_argument("--max-cases", type=int, default=None)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--strip-half-window-frames", type=int, default=6)
    p.add_argument("--strip-stride", type=int, default=1)
    p.add_argument("--video-start-time", type=str, default=None,
                   help="HH:MM:SS, default derived from video filename HHMMSS truncated to HH:MM:00")
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

        ours_index = build_direction_index(ours_rows)

        cases: list[DisagreeCase] = []
        for v in vendor_rows:
            if v.vehicle_class != args.vendor_class:
                continue
            if v.timestamp < window_start or v.timestamp > window_end:
                continue
            o, delta = nearest_same_direction(ours_index, v)
            if delta is None or abs(delta) > args.match_window_s:
                continue
            if o.vehicle_class != args.ours_class:
                continue
            frame_idx = timestamp_to_frame_idx(v.timestamp, video_t0, fps)
            cases.append(DisagreeCase(v, o, delta, frame_idx))
            if args.max_cases is not None and len(cases) >= args.max_cases:
                break

        index_rows = []
        saved = skipped = 0
        for case in cases:
            if case.frame_idx < 0 or case.frame_idx >= frame_count:
                skipped += 1
                warn(f"skipping {format_csv_timestamp(case.vendor.timestamp)} frame {case.frame_idx} out of [0, {frame_count - 1}]")
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
                f"{case.vendor.timestamp.strftime('%H%M%S')}_v{args.vendor_class}_o{args.ours_class}_"
                f"{case.vendor.video_file_num}_{case.vendor.frame_num}.png"
            )
            strip_path = args.out_dir / fname
            strip.save(strip_path)
            index_rows.append({
                "case_idx": str(saved + 1),
                "vendor_time": format_csv_timestamp(case.vendor.timestamp),
                "vendor_lane": case.vendor.lane,
                "vendor_class": case.vendor.vehicle_class,
                "ours_nearest_west_time": format_csv_timestamp(case.ours.timestamp),
                "ours_nearest_west_delta_s": format_delta(case.delta_s),
                "ours_nearest_west_lane": case.ours.lane + " cls=" + case.ours.vehicle_class,
                "strip_path": str(strip_path),
            })
            saved += 1
        write_index(args.out_dir / "index.csv", index_rows)
    finally:
        cap.release()

    print(f"vendor cls={args.vendor_class} matched ours cls={args.ours_class} within +/-{args.match_window_s}s: {len(cases)}")
    print(f"saved {saved} strips to {args.out_dir}")
    if skipped:
        print(f"skipped {skipped} out-of-range")


if __name__ == "__main__":
    main()
