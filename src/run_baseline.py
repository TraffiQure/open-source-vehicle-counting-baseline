"""Run YOLO11 plus tracking on a single video and emit PerVehicle CSV output."""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

from .counting import LineCounter
from .writer import write_pervehicle

COCO_TO_FHWA = {
    2: 2,
    3: 1,
    5: 4,
    7: 5,
}

# Modal-class fallbacks for string routing tokens. Used when cascade is off,
# or as cache-miss fallback when peak_crops has no entry for a track. Sources:
# memory/stage2_resolution_mismatch.md (art→9, su→5).
ROUTING_TOKEN_FALLBACK = {
    "axle_articulated": 9,
    "axle_single_unit": 5,
}


def load_class_map(path: Path | None) -> tuple[dict[int, int], dict[int, str]]:
    """Load fallback FHWA classes and routing decisions by detector class.

    JSON keys may be strings. Keys starting with '_' are treated as comments
    and ignored.

    Values may be:
      - int (1-13): direct FHWA class.
      - "drop" or negative int: exclude this detector class from counting.
      - "axle_articulated" / "axle_single_unit": include in class filter;
        FHWA class assigned by stage-2 cascade. When cascade is off (or on
        cache-miss), falls back to the modal class per ROUTING_TOKEN_FALLBACK.
    """
    if path is None:
        fallback = dict(COCO_TO_FHWA)
        routing = {detector_cls: "direct" for detector_cls in fallback}
        return fallback, routing
    raw = json.loads(path.read_text(encoding="utf-8"))
    fallback: dict[int, int] = {}
    routing: dict[int, str] = {}
    for k, v in raw.items():
        if str(k).startswith("_"):
            continue
        detector_cls = int(k)
        if isinstance(v, str):
            token = v.strip()
            if token == "drop":
                continue
            if token in ROUTING_TOKEN_FALLBACK:
                fallback[detector_cls] = ROUTING_TOKEN_FALLBACK[token]
                routing[detector_cls] = token
                continue
            try:
                iv = int(token)
            except ValueError as exc:
                raise ValueError(
                    f"Class map at {path}: unrecognized value {v!r} for key {k!r}. "
                    f"Expected int, 'drop', or one of {sorted(ROUTING_TOKEN_FALLBACK)}."
                ) from exc
        else:
            iv = int(v)
        if iv < 0:
            continue
        fallback[detector_cls] = iv
        routing[detector_cls] = "direct"
    if not fallback:
        raise ValueError(f"Class map at {path} is empty after filtering")
    return fallback, routing

TRACKER_CONFIGS = {
    "bytetrack": "bytetrack.yaml",
    "botsort": "botsort.yaml",
}

VIDEO_NAME_RE = re.compile(r"^(\d+)_(\d{14})__(\d+)\.avi$")
PEAK_CROP_EDGE_MARGIN_PX = 2.0


@dataclass(frozen=True, slots=True)
class VideoInfo:
    station: int
    report_start_time: datetime
    video_file_num: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--lines", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--model", type=str, default="yolo11s.pt")
    parser.add_argument(
        "--tracker",
        choices=sorted(TRACKER_CONFIGS),
        default="bytetrack",
    )
    parser.add_argument("--device", choices=["mps", "cpu", "cuda"], default="mps")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument(
        "--class-map",
        type=Path,
        default=None,
        help="Optional JSON {detector_class_id -> FHWA class or routing token}. Defaults to COCO_TO_FHWA.",
    )
    parser.add_argument(
        "--layer2-art-ckpt",
        type=Path,
        default=Path("runs/stage2_axle_articulated_v5_fixed/best.pt"),
    )
    parser.add_argument(
        "--layer2-su-ckpt",
        type=Path,
        default=Path("runs/stage2_axle_single_unit_v5_fixed/best.pt"),
    )
    parser.add_argument("--no-stage2", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.video.exists():
        raise FileNotFoundError(args.video)
    if not args.lines.exists():
        raise FileNotFoundError(args.lines)

    video_info = parse_video_filename(args.video.name)
    output_path = args.out or default_output_path(video_info)
    lines_payload = load_lines_payload(args.lines)
    counter = LineCounter(lines_payload["lines"])
    metadata = build_output_metadata(video_info, lines_payload)
    fallback_class_map, routing_map = load_class_map(args.class_map)
    cascade = None
    peak_crops: dict[int, dict[str, object]] | None = None
    stage2_class_counts: Counter[int] | None = None
    if not args.no_stage2:
        from .stage2_classifier import Stage2Cascade

        cascade = Stage2Cascade(
            layer2_art_ckpt=args.layer2_art_ckpt,
            layer2_su_ckpt=args.layer2_su_ckpt,
            device=args.device,
        )
        peak_crops = {}
        stage2_class_counts = Counter()

    from ultralytics import YOLO

    model = YOLO(args.model)
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {args.video}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise ValueError(f"Invalid FPS for video: {args.video}")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    progress = tqdm(total=total_frames or None, unit="frame")

    rows: list[dict[str, object]] = []
    per_line_counts: Counter[str] = Counter()
    frame_idx = 0
    start_time = time.perf_counter()

    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break

            timestamp = video_info.report_start_time + timedelta(seconds=frame_idx / fps)
            results = model.track(
                frame,
                tracker=TRACKER_CONFIGS[args.tracker],
                classes=sorted(fallback_class_map),
                conf=args.conf,
                persist=True,
                verbose=False,
                device=args.device,
            )

            if results:
                process_result(
                    results[0],
                    counter=counter,
                    frame_idx=frame_idx,
                    timestamp=timestamp,
                    video_file_num=video_info.video_file_num,
                    rows=rows,
                    per_line_counts=per_line_counts,
                    fallback_class_map=fallback_class_map,
                    routing_map=routing_map,
                    frame=frame,
                    peak_crops=peak_crops,
                    cascade=cascade,
                    stage2_class_counts=stage2_class_counts,
                )

            frame_idx += 1
            progress.update(1)
    finally:
        progress.close()
        capture.release()

    write_pervehicle(output_path, rows, metadata)

    elapsed = time.perf_counter() - start_time
    print(f"Processed frames: {frame_idx}")
    print(f"Crossings emitted: {len(rows)}")
    for line_name, count in sorted(per_line_counts.items()):
        print(f"{line_name}: {count}")
    if peak_crops is not None and stage2_class_counts is not None:
        truncated_peak_count = sum(
            1 for cached in peak_crops.values() if not bool(cached["fully_in_frame"])
        )
        print(
            f"Tracks with truncated peak crops: {truncated_peak_count} "
            f"(out of {len(peak_crops)})"
        )
        print(f"Stage-2 class distribution: {dict(sorted(stage2_class_counts.items()))}")
    print(f"Wall time (s): {elapsed:.2f}")


def parse_video_filename(filename: str) -> VideoInfo:
    match = VIDEO_NAME_RE.match(filename)
    if match is None:
        raise ValueError(
            "Video filename must match ^(\\d+)_(\\d{14})__(\\d+)\\.avi$"
        )
    # Filename HHMMSS encodes report-trigger time, but the burn-in clock
    # (true video t=0) starts at HH:MM:00. Truncate seconds to 0 so
    # CollectionTime = video_start + frame_idx/fps stays wall-clock correct.
    raw_start = datetime.strptime(match.group(2), "%Y%m%d%H%M%S")
    video_start = raw_start.replace(second=0)
    return VideoInfo(
        station=int(match.group(1)),
        report_start_time=video_start,
        video_file_num=int(match.group(3)),
    )


def default_output_path(video_info: VideoInfo) -> Path:
    date_token = video_info.report_start_time.strftime("%Y%m%d")
    return Path("outputs") / f"PerVehicle_{video_info.station}_{date_token}_opensource.csv"


def load_lines_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "lines" not in payload or not isinstance(payload["lines"], list):
        raise ValueError(f"Invalid lines JSON in {path}")
    if not payload["lines"]:
        raise ValueError(f"No lines defined in {path}")
    return payload


def build_output_metadata(
    video_info: VideoInfo, lines_payload: dict[str, Any]
) -> dict[str, object]:
    lines = lines_payload["lines"]
    direction_counts = Counter(str(line["direction"]) for line in lines)
    base_direction = str(lines_payload.get("base_direction") or lines[0]["direction"])
    if base_direction not in direction_counts:
        raise ValueError(f"Base direction {base_direction!r} is not present in lines.json")

    lane_count_base_dir = direction_counts[base_direction]
    lane_count_second_dir = sum(direction_counts.values()) - lane_count_base_dir
    return {
        "station": video_info.station,
        "site": str(lines_payload.get("site", "")),
        "location": str(lines_payload.get("location", "")),
        "report_start_time": video_info.report_start_time,
        "base_direction": base_direction,
        "lane_count_base_dir": lane_count_base_dir,
        "lane_count_second_dir": lane_count_second_dir,
    }


def process_result(
    result: Any,
    *,
    counter: LineCounter,
    frame_idx: int,
    timestamp: datetime,
    video_file_num: int,
    rows: list[dict[str, object]],
    per_line_counts: Counter[str],
    fallback_class_map: dict[int, int],
    routing_map: dict[int, str],
    frame: np.ndarray,
    peak_crops: dict[int, dict[str, object]] | None,
    cascade: Any | None,
    stage2_class_counts: Counter[int] | None,
) -> None:
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.id is None:
        return

    track_ids = boxes.id.int().cpu().tolist()
    class_ids = boxes.cls.int().cpu().tolist()
    boxes_xyxy = boxes.xyxy.cpu().tolist()
    detections = list(zip(track_ids, class_ids, boxes_xyxy))

    if peak_crops is not None:
        update_peak_crops(
            frame=frame,
            detections=detections,
            class_map=fallback_class_map,
            peak_crops=peak_crops,
        )

    for track_id, detector_class, bbox in detections:
        if detector_class not in fallback_class_map:
            continue
        x1, y1, x2, y2 = bbox
        center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        events = counter.update(track_id, center, frame_idx, timestamp, detector_class)
        for event in events:
            if cascade is not None:
                routing = routing_map.get(event.coco_class, "direct")
                cached = None if peak_crops is None else peak_crops.get(track_id)
                if routing in ("axle_articulated", "axle_single_unit") and cached is not None:
                    fhwa_class = int(cascade.classify(routing, cached["crop"]))
                else:
                    fhwa_class = fallback_class_map[event.coco_class]
            else:
                fhwa_class = fallback_class_map[event.coco_class]

            rows.append(
                {
                    "CollectionTime": event.timestamp,
                    "Direction": event.direction,
                    "Lane": event.lane,
                    "Class": fhwa_class,
                    "Speed": 0,
                    "videoFileNum": video_file_num,
                    "FrameNum": event.frame_idx,
                    "ImageFileName": "",
                }
            )
            per_line_counts[event.line_name] += 1
            if stage2_class_counts is not None:
                stage2_class_counts[fhwa_class] += 1


def update_peak_crops(
    *,
    frame: np.ndarray,
    detections: list[tuple[int, int, list[float]]],
    class_map: dict[int, int],
    peak_crops: dict[int, dict[str, object]],
) -> None:
    frame_h, frame_w = frame.shape[:2]

    for track_id, detector_class, bbox in detections:
        if detector_class not in class_map:
            continue

        x1, y1, x2, y2 = bbox
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if area <= 0:
            continue

        x1i = max(0, min(frame_w, int(x1)))
        y1i = max(0, min(frame_h, int(y1)))
        x2i = max(0, min(frame_w, int(x2)))
        y2i = max(0, min(frame_h, int(y2)))
        if x2i <= x1i or y2i <= y1i:
            continue

        fully_in_frame = (
            x1 >= PEAK_CROP_EDGE_MARGIN_PX
            and y1 >= PEAK_CROP_EDGE_MARGIN_PX
            and x2 <= frame_w - PEAK_CROP_EDGE_MARGIN_PX
            and y2 <= frame_h - PEAK_CROP_EDGE_MARGIN_PX
        )
        cached = peak_crops.get(track_id)
        should_replace = False
        if cached is None:
            should_replace = True
        elif fully_in_frame:
            should_replace = (not bool(cached["fully_in_frame"])) or area > float(
                cached["area"]
            )
        elif not bool(cached["fully_in_frame"]) and area > float(cached["area"]):
            should_replace = True

        if not should_replace:
            continue

        peak_crops[track_id] = {
            "area": area,
            "crop": frame[y1i:y2i, x1i:x2i].copy(),
            "fully_in_frame": fully_in_frame,
        }


if __name__ == "__main__":
    main()
