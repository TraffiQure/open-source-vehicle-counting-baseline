from __future__ import annotations

import argparse
import csv
import re
import sys
from bisect import bisect_left
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont


INDEX_FIELDS = [
    "case_idx",
    "vendor_time",
    "vendor_lane",
    "vendor_class",
    "ours_nearest_west_time",
    "ours_nearest_west_delta_s",
    "ours_nearest_west_lane",
    "strip_path",
]
TIMESTAMP_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y/%m/%d %H:%M:%S",
    "%Y/%m/%d  %H:%M:%S",
)
TILE_MAX_HEIGHT = 220
TEXT_MARGIN = 6
LANCZOS = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
VIDEO_FILENAME_RE = re.compile(r"_(?P<date>\d{8})(?P<time>\d{6})__")


@dataclass(frozen=True, slots=True)
class CrossingRecord:
    timestamp: datetime
    direction: str
    lane: str
    vehicle_class: str
    video_file_num: str
    frame_num: str


@dataclass(frozen=True, slots=True)
class MismatchCase:
    vendor: CrossingRecord
    ours_nearest: CrossingRecord | None
    ours_delta_s: float | None
    frame_idx: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ours", type=Path, required=True)
    parser.add_argument("--vendor", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--video-start-time", help="HH:MM:SS; default derived from --video filename")
    parser.add_argument("--direction", required=True)
    parser.add_argument("--lane", type=int, required=True)
    parser.add_argument("--window-s", type=float, default=5.0)
    parser.add_argument("--max-cases", type=int, default=20)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--strip-half-window-frames", type=int, default=6)
    parser.add_argument("--strip-stride", type=int, default=1)
    return parser.parse_args()


def parse_timestamp(raw: str) -> datetime | None:
    value = raw.strip()
    if not value:
        return None
    for fmt in TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def direction_key(value: str) -> str:
    return value.strip().casefold()


def ensure_output_dir(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise SystemExit(f"error: --out-dir exists and is not a directory: {path}")
        if any(path.iterdir()):
            raise SystemExit(
                f"error: --out-dir already exists and is not empty: {path}\n"
                "choose a different --out-dir or remove the existing contents"
            )
        return
    path.mkdir(parents=True, exist_ok=False)


def load_ours(path: Path, video_path: Path | None = None) -> list[CrossingRecord]:
    """Load ours PerVehicle CSV. The video_path arg is accepted for
    backwards compatibility with callers — it used to be needed for
    timestamp offset correction (ours pipeline used to anchor t=0 at
    filename HHMMSS instead of HH:MM:00). The anchor bug is now fixed
    in src/run_baseline.parse_video_filename, so timestamps from any
    CSV produced after 2026-04-27 are wall-clock-correct and need no
    correction. The arg is ignored."""
    del video_path  # unused; kept for caller signature stability
    rows: list[CrossingRecord] = []
    with path.open(newline="", encoding="utf-8") as handle:
        header: list[str] | None = None
        for line in handle:
            if line.startswith("CollectionTime"):
                header = [column.strip() for column in line.strip().split(",")]
                break
        if header is None:
            raise ValueError(f"Missing CollectionTime header in {path}")

        reader = csv.DictReader(handle, fieldnames=header)
        for row in reader:
            timestamp = parse_timestamp(row.get("CollectionTime", ""))
            if timestamp is None:
                continue
            rows.append(
                CrossingRecord(
                    timestamp=timestamp,
                    direction=(row.get("Direction") or "").strip(),
                    lane=(row.get("Lane") or "").strip(),
                    vehicle_class=(row.get("Class") or "").strip(),
                    video_file_num=(row.get("videoFileNum") or "").strip(),
                    frame_num=(row.get("FrameNum") or "").strip(),
                )
            )
    if not rows:
        raise ValueError(f"No data rows in {path}")
    return rows


def load_vendor(path: Path) -> list[CrossingRecord]:
    rows: list[CrossingRecord] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("CollectionTime"):
                break
        else:
            raise ValueError(f"Missing CollectionTime header in {path}")

        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            parsed = parse_vendor_row(row)
            if parsed is not None:
                rows.append(parsed)
    if not rows:
        raise ValueError(f"No vendor data rows in {path}")
    return rows


def parse_vendor_row(row: list[str]) -> CrossingRecord | None:
    values = [value.strip() for value in row]
    if len(values) >= 9:
        timestamp = parse_timestamp(values[0])
        if timestamp is None:
            return None
        return CrossingRecord(
            timestamp=timestamp,
            direction=values[2],
            lane=values[3],
            vehicle_class=values[4],
            video_file_num=values[6],
            frame_num=values[7],
        )
    if len(values) >= 8:
        timestamp = parse_timestamp(values[0])
        if timestamp is None:
            return None
        return CrossingRecord(
            timestamp=timestamp,
            direction=values[1],
            lane=values[2],
            vehicle_class=values[3],
            video_file_num=values[5],
            frame_num=values[6],
        )
    return None


def build_direction_index(
    rows: list[CrossingRecord],
) -> dict[str, tuple[list[datetime], list[CrossingRecord]]]:
    grouped: dict[str, list[CrossingRecord]] = {}
    for row in rows:
        grouped.setdefault(direction_key(row.direction), []).append(row)

    index: dict[str, tuple[list[datetime], list[CrossingRecord]]] = {}
    for key, items in grouped.items():
        items.sort(key=lambda item: item.timestamp)
        index[key] = ([item.timestamp for item in items], items)
    return index


def nearest_same_direction(
    index: dict[str, tuple[list[datetime], list[CrossingRecord]]],
    vendor_row: CrossingRecord,
) -> tuple[CrossingRecord | None, float | None]:
    timestamps, rows = index.get(direction_key(vendor_row.direction), ([], []))
    if not timestamps:
        return None, None

    position = bisect_left(timestamps, vendor_row.timestamp)
    candidates: list[CrossingRecord] = []
    if position < len(rows):
        candidates.append(rows[position])
    if position > 0:
        candidates.append(rows[position - 1])
    if not candidates:
        return None, None

    nearest = min(
        candidates,
        key=lambda row: abs((row.timestamp - vendor_row.timestamp).total_seconds()),
    )
    delta_s = (nearest.timestamp - vendor_row.timestamp).total_seconds()
    return nearest, delta_s


def parse_video_start_clock(value: str) -> time:
    try:
        return datetime.strptime(value, "%H:%M:%S").time()
    except ValueError as exc:
        raise SystemExit(
            f"error: invalid --video-start-time {value!r}; expected HH:MM:SS"
        ) from exc


def derive_default_video_start_clock(video_path: Path) -> time:
    match = VIDEO_FILENAME_RE.search(video_path.name)
    if match is None:
        raise SystemExit(
            f"error: could not derive --video-start-time from video filename {video_path.name!r}; "
            "pass --video-start-time HH:MM:SS explicitly"
        )
    hhmmss = match.group("time")
    try:
        parsed = datetime.strptime(hhmmss, "%H%M%S").time()
    except ValueError as exc:  # pragma: no cover
        raise SystemExit(
            f"error: could not parse start time from video filename {video_path.name!r}"
        ) from exc
    return parsed.replace(second=0)


def resolve_video_start_dt(
    video_path: Path,
    reference_dt: datetime,
    explicit_start_time: str | None,
) -> datetime:
    clock = (
        parse_video_start_clock(explicit_start_time)
        if explicit_start_time
        else derive_default_video_start_clock(video_path)
    )
    return datetime.combine(reference_dt.date(), clock)


def timestamp_to_frame_idx(
    event_time: datetime,
    video_start_dt: datetime,
    fps: float,
) -> int:
    return int(round((event_time - video_start_dt).total_seconds() * fps))


def collect_mismatches(
    ours: list[CrossingRecord],
    vendor: list[CrossingRecord],
    direction: str,
    lane: int,
    window_s: float,
    max_cases: int,
    video_start_dt: datetime,
    fps: float,
) -> tuple[list[MismatchCase], datetime, datetime]:
    window_start = min(row.timestamp for row in ours)
    window_end = max(row.timestamp for row in ours)
    lane_text = str(lane)
    direction_normalized = direction_key(direction)
    ours_index = build_direction_index(ours)

    mismatches: list[MismatchCase] = []
    for vendor_row in vendor:
        if direction_key(vendor_row.direction) != direction_normalized:
            continue
        if vendor_row.lane != lane_text:
            continue
        if vendor_row.timestamp < window_start or vendor_row.timestamp > window_end:
            continue

        ours_nearest, delta_s = nearest_same_direction(ours_index, vendor_row)
        if delta_s is not None and abs(delta_s) <= window_s:
            continue

        frame_idx = timestamp_to_frame_idx(vendor_row.timestamp, video_start_dt, fps)
        mismatches.append(
            MismatchCase(
                vendor=vendor_row,
                ours_nearest=ours_nearest,
                ours_delta_s=delta_s,
                frame_idx=frame_idx,
            )
        )
        if len(mismatches) >= max_cases:
            break

    return mismatches, window_start, window_end


def fit_tile(frame: Image.Image) -> Image.Image:
    if frame.height <= TILE_MAX_HEIGHT:
        return frame
    scale = TILE_MAX_HEIGHT / frame.height
    width = max(1, int(round(frame.width * scale)))
    return frame.resize((width, TILE_MAX_HEIGHT), LANCZOS)


def make_placeholder(width: int, height: int) -> Image.Image:
    return Image.new("RGB", (width, height), color=(24, 24, 24))


def annotate_tile(
    frame: Image.Image,
    frame_index: int,
    rel_seconds: float,
    is_center: bool,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
) -> Image.Image:
    tile = fit_tile(frame).convert("RGB")
    draw = ImageDraw.Draw(tile)
    text = f"f={frame_index}\n{rel_seconds:+.2f}s"
    text_box = draw.multiline_textbbox((0, 0), text, font=font, spacing=2)
    box_width = text_box[2] - text_box[0]
    box_height = text_box[3] - text_box[1]
    draw.rectangle(
        (
            TEXT_MARGIN - 2,
            TEXT_MARGIN - 2,
            TEXT_MARGIN + box_width + 2,
            TEXT_MARGIN + box_height + 2,
        ),
        fill=(0, 0, 0),
    )
    draw.multiline_text(
        (TEXT_MARGIN, TEXT_MARGIN),
        text,
        fill=(255, 255, 255),
        font=font,
        spacing=2,
    )

    border_color = (255, 0, 0) if is_center else (160, 160, 160)
    border_width = 4 if is_center else 1
    for offset in range(border_width):
        draw.rectangle(
            (offset, offset, tile.width - 1 - offset, tile.height - 1 - offset),
            outline=border_color,
        )
    return tile


def read_frame(
    capture: cv2.VideoCapture,
    frame_index: int,
) -> Image.Image | None:
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame_bgr = capture.read()
    if not ok:
        return None
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


def render_strip(
    capture: cv2.VideoCapture,
    center_frame_idx: int,
    fps: float,
    frame_count: int,
    half_window_frames: int,
    stride: int,
) -> Image.Image:
    indices = [
        center_frame_idx + (offset * stride)
        for offset in range(-half_window_frames, half_window_frames + 1)
    ]

    font = ImageFont.load_default()
    sampled_frames: list[tuple[int, Image.Image | None]] = []
    fallback_width = 320
    fallback_height = 180

    for frame_index in indices:
        if 0 <= frame_index < frame_count:
            frame = read_frame(capture, frame_index)
            sampled_frames.append((frame_index, frame))
            if frame is not None:
                fallback_width = frame.width
                fallback_height = frame.height
        else:
            sampled_frames.append((frame_index, None))

    tiles: list[Image.Image] = []
    center_offset = len(indices) // 2
    for position, (frame_index, frame) in enumerate(sampled_frames):
        if frame is None:
            frame = make_placeholder(fallback_width, fallback_height)
        rel_seconds = (frame_index - center_frame_idx) / fps
        tiles.append(
            annotate_tile(
                frame=frame,
                frame_index=frame_index,
                rel_seconds=rel_seconds,
                is_center=position == center_offset,
                font=font,
            )
        )

    strip_width = sum(tile.width for tile in tiles)
    strip_height = max(tile.height for tile in tiles)
    strip = Image.new("RGB", (strip_width, strip_height), color=(12, 12, 12))

    x_offset = 0
    for tile in tiles:
        y_offset = (strip_height - tile.height) // 2
        strip.paste(tile, (x_offset, y_offset))
        x_offset += tile.width
    return strip


def write_index(
    path: Path,
    rows: list[dict[str, str]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=INDEX_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def format_csv_timestamp(value: datetime | None) -> str:
    if value is None:
        return ""
    return value.strftime("%Y-%m-%d %H:%M:%S")


def format_delta(delta_s: float | None) -> str:
    if delta_s is None:
        return ""
    return f"{delta_s:.3f}"


def warn(message: str) -> None:
    print(f"warning: {message}", file=sys.stderr)


def main() -> None:
    args = parse_args()
    if args.max_cases < 1:
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

        mismatches, window_start, window_end = collect_mismatches(
            ours=ours_rows,
            vendor=vendor_rows,
            direction=args.direction,
            lane=args.lane,
            window_s=args.window_s,
            max_cases=args.max_cases,
            video_start_dt=video_start_dt,
            fps=fps,
        )

        index_rows: list[dict[str, str]] = []
        saved_count = 0
        skipped_count = 0

        for mismatch in mismatches:
            if mismatch.frame_idx < 0 or mismatch.frame_idx >= frame_count:
                skipped_count += 1
                warn(
                    "skipping vendor timestamp "
                    f"{format_csv_timestamp(mismatch.vendor.timestamp)} "
                    f"because frame index {mismatch.frame_idx} is outside [0, {frame_count - 1}]"
                )
                continue

            strip = render_strip(
                capture=capture,
                center_frame_idx=mismatch.frame_idx,
                fps=fps,
                frame_count=frame_count,
                half_window_frames=args.strip_half_window_frames,
                stride=args.strip_stride,
            )
            file_name = (
                f"{mismatch.vendor.timestamp.strftime('%H%M%S')}_track_"
                f"{mismatch.vendor.video_file_num}_{mismatch.vendor.frame_num}.png"
            )
            strip_path = args.out_dir / file_name
            strip.save(strip_path)

            index_rows.append(
                {
                    "case_idx": str(saved_count + 1),
                    "vendor_time": format_csv_timestamp(mismatch.vendor.timestamp),
                    "vendor_lane": mismatch.vendor.lane,
                    "vendor_class": mismatch.vendor.vehicle_class,
                    "ours_nearest_west_time": format_csv_timestamp(
                        mismatch.ours_nearest.timestamp if mismatch.ours_nearest else None
                    ),
                    "ours_nearest_west_delta_s": format_delta(mismatch.ours_delta_s),
                    "ours_nearest_west_lane": (
                        mismatch.ours_nearest.lane if mismatch.ours_nearest else ""
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
        f"found {len(mismatches)} mismatch candidates for "
        f"{args.direction} lane {args.lane} within ±{args.window_s:g}s"
    )
    print(f"saved {saved_count} strips to {args.out_dir}")
    if skipped_count:
        print(f"skipped {skipped_count} out-of-range cases", file=sys.stderr)
    print(f"wrote index: {args.out_dir / 'index.csv'}")


if __name__ == "__main__":
    main()
