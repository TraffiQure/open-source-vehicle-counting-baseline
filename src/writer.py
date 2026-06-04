"""Write Vendor L PerVehicle CSV output."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

HEADER_LINE = (
    "CollectionTime, Direction, Lane, Class, Speed, videoFileNum, FrameNum, "
    "ImageFileName"
)

REQUIRED_METADATA = [
    "station",
    "site",
    "location",
    "report_start_time",
    "base_direction",
    "lane_count_base_dir",
    "lane_count_second_dir",
]


def write_pervehicle(
    path: Path | str, rows: Sequence[Mapping[str, object]], metadata: Mapping[str, object]
) -> None:
    output_path = Path(path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    missing = [key for key in REQUIRED_METADATA if key not in metadata]
    if missing:
        raise ValueError(f"Missing required metadata for PerVehicle output: {missing}")

    report_start_time = _coerce_datetime(metadata["report_start_time"])
    ordered_rows = sorted(
        rows,
        key=lambda row: (
            _coerce_datetime(row["CollectionTime"]),
            int(row["FrameNum"]),
            str(row["Direction"]),
            int(row["Lane"]),
        ),
    )

    lines = [
        f"Data File:,{output_path},   AI Count Version:, opensource-v1",
        f"Station Number:,{metadata['station']}",
        f"Site:,{metadata['site']}",
        f"Location:,{metadata['location']}",
        f"Report Start Time:,{report_start_time.strftime('%Y/%m/%d  %H:%M:%S')}",
        f"Base Direction:,{metadata['base_direction']}",
        "latitude:,0",
        "longitude:,0",
        f"Lane number Base direction:,{metadata['lane_count_base_dir']}",
        f"Lane number second direction:,{metadata['lane_count_second_dir']}",
        "Classification Type:, 13 Class Classification",
        "Unit Serial Number:,opensource-v1",
        "",
        HEADER_LINE,
    ]

    for row in ordered_rows:
        collection_time = _coerce_datetime(row["CollectionTime"]).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        speed = int(row["Speed"])
        frame_num = int(row["FrameNum"])
        lines.append(
            ",".join(
                [
                    collection_time,
                    str(row["Direction"]),
                    str(int(row["Lane"])),
                    str(int(row["Class"])),
                    f"{speed:03d}",
                    str(int(row["videoFileNum"])),
                    f"{frame_num:06d}",
                    "",
                ]
            )
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _coerce_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    raise ValueError(f"Expected datetime-compatible value, got {type(value)!r}")
