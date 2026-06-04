"""Load Vendor L CSV reports into normalized DataFrames."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pandas as pd

PER_VEHICLE_COLUMNS = [
    "CollectionTime",
    "Direction",
    "Lane",
    "Class",
    "Speed",
    "videoFileNum",
    "FrameNum",
    "ImageFileName",
]

PER_VEHICLE_METADATA_MAP = {
    "Station Number:": "station_number",
    "Site:": "site",
    "Location:": "location",
    "Report Start Time:": "report_start_time",
    "Base Direction:": "base_direction",
    "latitude:": "latitude",
    "longitude:": "longitude",
    "Lane number Base direction:": "lane_count_base_dir",
    "Lane number second direction:": "lane_count_second_dir",
    "Classification Type:": "classification_type",
    "Unit Serial Number:": "unit_serial_number",
}

MS2_DIRECTION_MAP = {
    "EB": "East",
    "WB": "West",
    "NB": "North",
    "SB": "South",
}


def read_pervehicle(path: Path, scenario: str | None = None) -> pd.DataFrame:
    csv_path = Path(path)
    rows = _read_csv_rows(csv_path)
    if len(rows) < 15:
        raise ValueError(f"PerVehicle file is too short: {csv_path}")

    metadata = _parse_pervehicle_metadata(rows[:12])
    first_data_row = _first_nonempty_row(rows[14:])
    field_count = len(first_data_row)
    if field_count == 9:
        names = ["DuplicateCollectionTime", *PER_VEHICLE_COLUMNS]
    elif field_count == 8:
        names = PER_VEHICLE_COLUMNS
    else:
        raise ValueError(
            f"Unexpected PerVehicle row width {field_count} in {csv_path}"
        )

    try:
        df = pd.read_csv(
            csv_path,
            skiprows=14,
            header=None,
            names=names,
            keep_default_na=False,
            skipinitialspace=True,
        )
    except Exception as exc:  # pragma: no cover - delegated parse errors
        raise ValueError(f"Could not parse PerVehicle CSV: {csv_path}") from exc

    df = _drop_empty_rows(df)
    if field_count == 9:
        df = df.drop(columns=["DuplicateCollectionTime"])

    _require_columns(df, PER_VEHICLE_COLUMNS, csv_path)

    try:
        df["CollectionTime"] = pd.to_datetime(
            df["CollectionTime"], format="%Y-%m-%d %H:%M:%S"
        )
        df["Lane"] = _to_int_series(df["Lane"])
        df["Class"] = _to_int_series(df["Class"])
        df["Speed"] = _to_int_series(df["Speed"])
        df["videoFileNum"] = _to_int_series(df["videoFileNum"])
        df["FrameNum"] = _to_int_series(df["FrameNum"])
        df["ImageFileName"] = df["ImageFileName"].astype(str)
    except Exception as exc:  # pragma: no cover - delegated parse errors
        raise ValueError(f"Invalid PerVehicle row data in {csv_path}") from exc

    df["scenario"] = scenario
    df["station_number"] = int(metadata["station_number"])
    df = df[PER_VEHICLE_COLUMNS + ["scenario", "station_number"]]
    df.attrs = metadata
    return df


def read_binned_class(path: Path, scenario: str | None = None) -> pd.DataFrame:
    csv_path = Path(path)
    rows = _read_csv_rows(csv_path)
    if len(rows) < 3:
        raise ValueError(f"BinnedClass file is too short: {csv_path}")

    unit_serial_number = _clean_value(rows[0][1] if len(rows[0]) > 1 else "")

    try:
        df = pd.read_csv(
            csv_path,
            skiprows=1,
            header=0,
            keep_default_na=False,
            skipinitialspace=True,
        )
    except Exception as exc:  # pragma: no cover - delegated parse errors
        raise ValueError(f"Could not parse BinnedClass CSV: {csv_path}") from exc

    df = _drop_empty_rows(df)
    df = df.rename(columns=lambda value: str(value).strip())

    class_columns = [f"class{index}" for index in range(1, 14)]
    expected_columns = [
        "StationNum",
        "Site",
        "Location",
        "Direction",
        "Lane",
        "Time",
        "SumTotal",
        *class_columns,
    ]
    _require_columns(df, expected_columns, csv_path)

    try:
        df["StationNum"] = _to_int_series(df["StationNum"])
        df["Lane"] = _to_int_series(df["Lane"])
        df["Time"] = pd.to_datetime(df["Time"], format="%Y/%m/%d %H:%M")
        df["SumTotal"] = _to_int_series(df["SumTotal"])
        for column in class_columns:
            df[column] = _to_int_series(df[column])
    except Exception as exc:  # pragma: no cover - delegated parse errors
        raise ValueError(f"Invalid BinnedClass row data in {csv_path}") from exc

    df["scenario"] = scenario
    df = df[expected_columns + ["scenario"]]
    df.attrs = {"unit_serial_number": unit_serial_number}
    return df


def read_binned_volume(path: Path, scenario: str | None = None) -> pd.DataFrame:
    csv_path = Path(path)
    rows = _read_csv_rows(csv_path)
    if len(rows) < 6:
        raise ValueError(f"BinnedVolume file is too short: {csv_path}")

    attrs = _parse_generic_metadata(rows[:4])

    try:
        df = pd.read_csv(
            csv_path,
            skiprows=4,
            header=0,
            keep_default_na=False,
            skipinitialspace=True,
        )
    except Exception as exc:  # pragma: no cover - delegated parse errors
        raise ValueError(f"Could not parse BinnedVolume CSV: {csv_path}") from exc

    df = _drop_empty_rows(df)
    df = df.rename(columns=lambda value: str(value).strip())
    df = df.drop(
        columns=[
            column
            for column in df.columns
            if column == "" or str(column).startswith("Unnamed:")
        ]
    )
    if "Count Total" not in df.columns:
        raise ValueError(f"Missing Count Total column in {csv_path}")
    df = df.rename(columns={"Count Total": "Count_Total"})

    expected_columns = ["Time", "Count_Total"]
    _require_columns(df, expected_columns, csv_path)
    lane_columns = [
        column for column in df.columns if column not in {"Time", "Count_Total"}
    ]
    if not lane_columns:
        raise ValueError(f"No lane columns found in {csv_path}")

    try:
        df["Time"] = pd.to_datetime(df["Time"], format="%Y/%m/%d %H:%M")
        df["Count_Total"] = _to_int_series(df["Count_Total"])
        for column in lane_columns:
            df[column] = _to_int_series(df[column])
    except Exception as exc:  # pragma: no cover - delegated parse errors
        raise ValueError(f"Invalid BinnedVolume row data in {csv_path}") from exc

    df["scenario"] = scenario
    df = df[["Time", "Count_Total", *lane_columns, "scenario"]]
    df.attrs = attrs
    return df


def read_ms2_event(path: Path, scenario: str | None = None) -> pd.DataFrame:
    csv_path = Path(path)
    rows = _read_csv_rows(csv_path)
    header_index = _find_header_index(rows, "Vehicle")
    if header_index is None:
        raise ValueError(f"Could not find MS2 header row in {csv_path}")

    attrs = _parse_generic_metadata(rows[:header_index])

    try:
        df = pd.read_csv(
            csv_path,
            skiprows=header_index,
            header=0,
            keep_default_na=False,
            skipinitialspace=True,
        )
    except Exception as exc:  # pragma: no cover - delegated parse errors
        raise ValueError(f"Could not parse MS2 CSV: {csv_path}") from exc

    df = _drop_empty_rows(df)
    df = df.rename(columns=lambda value: str(value).strip())
    rename_map = {
        "Date / Time": "DateTime",
        "Speed(mph)": "Speed_mph",
        "Spacing(ft.)": "Spacing_ft",
    }
    df = df.rename(columns=rename_map)

    expected_columns = [
        "Vehicle",
        "Channel",
        "DateTime",
        "Axles",
        "Rule",
        "Class",
        "Speed_mph",
        "Gap",
        "Headway",
        "Acceleration",
        "Spacing_ft",
    ]
    _require_columns(df, expected_columns, csv_path)

    try:
        df["Vehicle"] = _to_int_series(df["Vehicle"])
        df["DateTime"] = pd.to_datetime(
            df["DateTime"], format="%m/%d/%Y %H:%M:%S.%f"
        )
        df["Axles"] = _to_int_series(df["Axles"])
        df["Rule"] = _to_int_series(df["Rule"])
        df["Class"] = _to_int_series(df["Class"])
        df["Speed_mph"] = pd.to_numeric(df["Speed_mph"].astype(str).str.strip())
        df["Gap"] = _to_int_series(df["Gap"])
        df["Headway"] = df["Headway"].astype(str)
        df["Acceleration"] = pd.to_numeric(df["Acceleration"].astype(str).str.strip())
        df["Spacing_ft"] = _to_int_series(df["Spacing_ft"])
    except Exception as exc:  # pragma: no cover - delegated parse errors
        raise ValueError(f"Invalid MS2 row data in {csv_path}") from exc

    lane_direction = df["Channel"].map(_parse_channel)
    df["Lane"] = [lane for lane, _ in lane_direction]
    df["Direction"] = [direction for _, direction in lane_direction]
    df["scenario"] = scenario
    df = df[expected_columns + ["Lane", "Direction", "scenario"]]
    df.attrs = attrs
    return df


def _read_csv_rows(path: Path) -> list[list[str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.reader(handle))


def _parse_pervehicle_metadata(rows: list[list[str]]) -> dict[str, Any]:
    if len(rows) != 12:
        raise ValueError("PerVehicle metadata must contain 12 lines")

    first_row = rows[0]
    if len(first_row) < 4:
        raise ValueError("PerVehicle metadata first row is malformed")

    metadata: dict[str, Any] = {
        "ai_count_version": _clean_value(first_row[3]),
    }

    for row in rows:
        if not row:
            continue
        label = _clean_value(row[0], strip_quotes=False)
        if label == "Data File:":
            continue
        key = PER_VEHICLE_METADATA_MAP.get(label)
        if key is None:
            raise ValueError(f"Unexpected PerVehicle metadata label: {label}")
        if len(row) < 2:
            raise ValueError(f"Missing value for PerVehicle metadata label: {label}")
        metadata[key] = _clean_value(row[1])

    metadata["station_number"] = int(metadata["station_number"])
    metadata["report_start_time"] = pd.to_datetime(
        metadata["report_start_time"], format="%Y/%m/%d  %H:%M:%S"
    )
    metadata["lane_count_base_dir"] = int(metadata["lane_count_base_dir"])
    metadata["lane_count_second_dir"] = int(metadata["lane_count_second_dir"])
    return metadata


def _parse_generic_metadata(rows: list[list[str]]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for row in rows:
        if not row or all(not cell.strip() for cell in row):
            continue
        key = _clean_value(row[0], strip_quotes=False)
        if not key:
            continue
        value = _clean_value(row[1]) if len(row) > 1 else ""
        metadata[_normalize_metadata_key(key)] = value
    if not metadata:
        raise ValueError("Metadata block is empty")
    return metadata


def _find_header_index(rows: list[list[str]], first_column: str) -> int | None:
    for index, row in enumerate(rows):
        if row and _clean_value(row[0], strip_quotes=False) == first_column:
            return index
    return None


def _first_nonempty_row(rows: list[list[str]]) -> list[str]:
    for row in rows:
        if any(cell.strip() for cell in row):
            return row
    raise ValueError("No data rows found")


def _drop_empty_rows(df: pd.DataFrame) -> pd.DataFrame:
    mask = ~(df.astype(str).apply(lambda row: row.str.strip().eq("").all(), axis=1))
    return df.loc[mask].copy()


def _require_columns(df: pd.DataFrame, columns: list[str], path: Path) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")


def _to_int_series(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.strip()
    return cleaned.astype(int)


def _parse_channel(channel: str) -> tuple[int, str]:
    parts = str(channel).strip().split("_", maxsplit=1)
    if len(parts) != 2:
        raise ValueError(f"Invalid MS2 channel value: {channel}")
    lane = int(parts[0])
    direction = MS2_DIRECTION_MAP.get(parts[1])
    if direction is None:
        raise ValueError(f"Unsupported MS2 direction code: {channel}")
    return lane, direction


def _normalize_metadata_key(label: str) -> str:
    cleaned = _clean_value(label, strip_quotes=False).rstrip(":")
    cleaned = cleaned.replace("/", " ").replace("-", " ").replace(".", "")
    return "_".join(cleaned.lower().split())


def _clean_value(value: str, strip_quotes: bool = True) -> str:
    cleaned = str(value).strip()
    if strip_quotes:
        cleaned = cleaned.strip("'").strip('"')
    return cleaned
