from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2

from dump_mismatch_frames import (
    CrossingRecord,
    build_direction_index,
    timestamp_to_frame_idx,
    ensure_output_dir,
    format_csv_timestamp,
    format_delta,
    load_ours,
    load_vendor,
    nearest_same_direction,
    render_strip,
    resolve_video_start_dt,
    warn,
    write_index,
)


@dataclass(frozen=True, slots=True)
class VendorClassCase:
    vendor: CrossingRecord
    ours_nearest_in_window: CrossingRecord | None
    ours_delta_s_in_window: float | None
    frame_idx: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ours", type=Path, required=True)
    parser.add_argument("--vendor", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--video-start-time", help="HH:MM:SS; default derived from --video filename")
    parser.add_argument("--vendor-classes", required=True)
    parser.add_argument("--match-window-s", type=float, default=5.0)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--strip-half-window-frames", type=int, default=6)
    parser.add_argument("--strip-stride", type=int, default=1)
    return parser.parse_args()


def parse_vendor_classes(raw: str) -> set[str]:
    classes = {value.strip() for value in raw.split(",") if value.strip()}
    if not classes:
        raise SystemExit("error: --vendor-classes must contain at least one class id")
    return classes


def collect_vendor_class_cases(
    ours: list[CrossingRecord],
    vendor: list[CrossingRecord],
    vendor_classes: set[str],
    match_window_s: float,
    max_cases: int | None,
    video_start_dt: datetime,
    fps: float,
) -> tuple[list[VendorClassCase], datetime, datetime]:
    window_start = min(row.timestamp for row in ours)
    window_end = max(row.timestamp for row in ours)
    ours_index = build_direction_index(ours)

    cases: list[VendorClassCase] = []
    for vendor_row in vendor:
        if vendor_row.vehicle_class not in vendor_classes:
            continue
        if vendor_row.timestamp < window_start or vendor_row.timestamp > window_end:
            continue

        ours_nearest, delta_s = nearest_same_direction(ours_index, vendor_row)
        if delta_s is None or abs(delta_s) > match_window_s:
            ours_nearest = None
            delta_s = None

        frame_idx = timestamp_to_frame_idx(vendor_row.timestamp, video_start_dt, fps)
        cases.append(
            VendorClassCase(
                vendor=vendor_row,
                ours_nearest_in_window=ours_nearest,
                ours_delta_s_in_window=delta_s,
                frame_idx=frame_idx,
            )
        )
        if max_cases is not None and len(cases) >= max_cases:
            break

    return cases, window_start, window_end


def main() -> None:
    args = parse_args()
    vendor_classes = parse_vendor_classes(args.vendor_classes)
    if args.max_cases is not None and args.max_cases < 1:
        raise SystemExit("error: --max-cases must be at least 1")
    if args.strip_half_window_frames < 0:
        raise SystemExit("error: --strip-half-window-frames must be >= 0")
    if args.strip_stride < 1:
        raise SystemExit("error: --strip-stride must be >= 1")

    ensure_output_dir(args.out_dir)

    ours_rows = load_ours(args.ours, args.video)
    vendor_rows = load_vendor(args.vendor)

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise SystemExit(f"error: failed to open video: {args.video}")

    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            raise SystemExit(f"error: failed to read FPS from video: {args.video}")
        frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        if frame_count <= 0:
            raise SystemExit(f"error: failed to read frame count from video: {args.video}")
        window_start = min(row.timestamp for row in ours_rows)
        video_start_dt = resolve_video_start_dt(
            args.video,
            reference_dt=window_start,
            explicit_start_time=args.video_start_time,
        )

        cases, window_start, window_end = collect_vendor_class_cases(
            ours=ours_rows,
            vendor=vendor_rows,
            vendor_classes=vendor_classes,
            match_window_s=args.match_window_s,
            max_cases=args.max_cases,
            video_start_dt=video_start_dt,
            fps=fps,
        )

        index_rows: list[dict[str, str]] = []
        saved_count = 0
        skipped_count = 0

        for case in cases:
            if case.frame_idx < 0 or case.frame_idx >= frame_count:
                skipped_count += 1
                warn(
                    "skipping vendor timestamp "
                    f"{format_csv_timestamp(case.vendor.timestamp)} "
                    f"because frame index {case.frame_idx} is outside [0, {frame_count - 1}]"
                )
                continue

            strip = render_strip(
                capture=capture,
                center_frame_idx=case.frame_idx,
                fps=fps,
                frame_count=frame_count,
                half_window_frames=args.strip_half_window_frames,
                stride=args.strip_stride,
            )
            file_name = (
                f"{case.vendor.timestamp.strftime('%H%M%S')}_cls{case.vendor.vehicle_class}_"
                f"{case.vendor.video_file_num}_{case.vendor.frame_num}.png"
            )
            strip_path = args.out_dir / file_name
            strip.save(strip_path)

            index_rows.append(
                {
                    "case_idx": str(saved_count + 1),
                    "vendor_time": format_csv_timestamp(case.vendor.timestamp),
                    "vendor_lane": case.vendor.lane,
                    "vendor_class": case.vendor.vehicle_class,
                    "ours_nearest_west_time": format_csv_timestamp(
                        case.ours_nearest_in_window.timestamp
                        if case.ours_nearest_in_window is not None
                        else None
                    ),
                    "ours_nearest_west_delta_s": format_delta(case.ours_delta_s_in_window),
                    "ours_nearest_west_lane": (
                        case.ours_nearest_in_window.lane
                        if case.ours_nearest_in_window is not None
                        else ""
                    ),
                    "strip_path": str(strip_path),
                }
            )
            saved_count += 1

        write_index(args.out_dir / "index.csv", index_rows)

    finally:
        capture.release()

    print(
        f"ours window: {format_csv_timestamp(window_start)} -> {format_csv_timestamp(window_end)}"
    )
    print(
        f"found {len(cases)} vendor crossings in classes "
        f"{','.join(sorted(vendor_classes))} within window"
    )
    print(f"saved {saved_count} strips to {args.out_dir}")
    if skipped_count:
        print(f"skipped {skipped_count} out-of-range cases")
    print(f"wrote index: {args.out_dir / 'index.csv'}")


if __name__ == "__main__":
    main()
