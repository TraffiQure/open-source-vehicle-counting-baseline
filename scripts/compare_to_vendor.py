"""Compare opensource PerVehicle CSVs against Vendor L PerVehicle, hour-matched.

For each opensource CSV (one per video, ~1h slice), find rows in vendor CSV
that fall in the same wall-clock hour and produce a comparison table:

  Total | per-direction | per-lane | class distribution | rare-class flags
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import datetime
from pathlib import Path


def parse_opensource(path: Path) -> tuple[datetime, datetime, list[dict]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for line in f:
            if line.startswith("CollectionTime"):
                header = [c.strip() for c in line.strip().split(",")]
                break
        reader = csv.DictReader(f, fieldnames=header)
        for row in reader:
            t_str = (row.get("CollectionTime") or "").strip()
            t = None
            for fmt in ("%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
                try:
                    t = datetime.strptime(t_str, fmt)
                    break
                except ValueError:
                    continue
            if t is None:
                try:
                    t = datetime.fromisoformat(t_str)
                except ValueError:
                    continue
            row["_t"] = t
            row["Direction"] = (row.get("Direction") or "").strip()
            row["Lane"] = (row.get("Lane") or "").strip()
            row["Class"] = (row.get("Class") or "").strip()
            rows.append(row)
    if not rows:
        raise ValueError(f"No data rows in {path}")
    return rows[0]["_t"], rows[-1]["_t"], rows


def parse_vendor(path: Path) -> list[dict]:
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for line in f:
            if line.startswith("CollectionTime"):
                break
        reader = csv.DictReader(
            f,
            fieldnames=[
                "CollectionTime", "CollectionTime2", "Direction", "Lane", "Class",
                "Speed", "videoFileNum", "FrameNum", "ImageFileName",
            ],
        )
        for row in reader:
            try:
                t = datetime.strptime(row["CollectionTime"].strip(), "%Y/%m/%d %H:%M:%S")
            except (ValueError, KeyError, AttributeError):
                continue
            row["_t"] = t
            row["Direction"] = row["Direction"].strip()
            row["Lane"] = row["Lane"].strip()
            row["Class"] = row["Class"].strip()
            rows.append(row)
    return rows


def stats(rows: list[dict]) -> dict:
    total = len(rows)
    by_dir = Counter(r["Direction"] for r in rows)
    by_lane = Counter((r["Direction"], r["Lane"]) for r in rows)
    by_class = Counter(int(r["Class"]) for r in rows)
    return {
        "total": total,
        "by_dir": dict(by_dir),
        "by_lane": {f"{d}_{l}": n for (d, l), n in sorted(by_lane.items())},
        "by_class": dict(sorted(by_class.items())),
    }


def fmt_dist(d: dict) -> str:
    return "{" + ", ".join(f"{k}:{v}" for k, v in sorted(d.items())) + "}"


def compare_one(
    opensource_csv: Path, vendor_csv: Path, label: str
) -> None:
    t_start, t_end, ours = parse_opensource(opensource_csv)
    vendor_all = parse_vendor(vendor_csv)
    vendor_window = [r for r in vendor_all if t_start <= r["_t"] <= t_end]

    s_ours = stats(ours)
    s_vend = stats(vendor_window)

    print(f"\n## {label}")
    print(f"window: {t_start}  →  {t_end}")
    print(f"opensource csv: {opensource_csv.name}")
    print(f"vendor csv: {vendor_csv.name}")
    print()
    print(f"|                | ours | vendor | delta |")
    print(f"|----------------|------|--------|-------|")
    print(f"| Total          | {s_ours['total']} | {s_vend['total']} | {s_ours['total']-s_vend['total']:+d} |")
    all_lanes = sorted(set(s_ours["by_lane"]) | set(s_vend["by_lane"]))
    for lane in all_lanes:
        o = s_ours["by_lane"].get(lane, 0)
        v = s_vend["by_lane"].get(lane, 0)
        print(f"| {lane:<14} | {o} | {v} | {o-v:+d} |")
    print()
    print(f"class dist (ours)  : {fmt_dist(s_ours['by_class'])}")
    print(f"class dist (vendor): {fmt_dist(s_vend['by_class'])}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True,
                    help="triplets: LABEL,OPEN_CSV,VENDOR_CSV")
    args = ap.parse_args()
    for triple in args.pairs:
        label, open_csv, vendor_csv = triple.split(",")
        compare_one(Path(open_csv), Path(vendor_csv), label)


if __name__ == "__main__":
    main()
