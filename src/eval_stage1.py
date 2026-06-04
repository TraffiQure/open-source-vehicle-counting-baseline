"""Phase-2 Stage-1 evaluation harness.

Compares one or more open-source baseline PerVehicle CSVs against the Vendor L
PerVehicle CSV for the same video, with per-lane and 15-minute breakdowns.

Typical usage after stage-1 fine-tune finishes:

    python -m src.eval_stage1 \
        --vendor Arterial/PerVehicle_85_20260308.csv \
        --baseline outputs/PerVehicle_85_20260308_opensource_v3.csv \
        --baseline outputs/PerVehicle_85_20260308_opensource_v7_miotcd.csv \
        --labels "Vendor,v3 (YOLO11s COCO),v7 (YOLO11s MIO-TCD fine-tune)" \
        --bin-minutes 15 \
        --out outputs/eval_stage1_summary.csv

Emits:
    - stdout: a formatted per-lane hour total table + per-bin tables
    - --out CSV (optional): long-format {source, lane, bin_start, count} for
      downstream plotting.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .vendor_loader import read_pervehicle


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vendor", type=Path, required=True,
                   help="Vendor PerVehicle CSV (reference)")
    p.add_argument("--baseline", type=Path, action="append", required=True,
                   help="Baseline PerVehicle CSV (may be repeated)")
    p.add_argument("--labels", type=str, default=None,
                   help="Comma-separated column labels: 'Vendor,<baseline1>,<baseline2>,...'. "
                        "Defaults to file stems.")
    p.add_argument("--bin-minutes", type=int, default=15)
    p.add_argument("--start", type=str, default=None,
                   help="ISO datetime; filter vendor+baselines to >= this. "
                        "If omitted, inferred from the earliest baseline timestamp.")
    p.add_argument("--end", type=str, default=None,
                   help="ISO datetime; filter vendor+baselines to < this. "
                        "If omitted, inferred from the latest baseline timestamp + 1s.")
    p.add_argument("--out", type=Path, default=None,
                   help="Optional long-format CSV with {source, lane, bin_start, count}")
    return p.parse_args()


def load(path: Path, label: str) -> pd.DataFrame:
    df = read_pervehicle(path)
    if "CollectionTime" not in df.columns:
        raise ValueError(f"{path}: missing CollectionTime column")
    df = df.copy()
    df["CollectionTime"] = pd.to_datetime(df["CollectionTime"])
    df["source"] = label
    df["lane"] = df["Direction"].astype(str) + "_" + df["Lane"].astype(str).str.zfill(2)
    return df[["source", "CollectionTime", "Direction", "Lane", "lane", "Class"]]


def hour_totals(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Wide table: rows=lane, cols=source, values=total count."""
    all_df = pd.concat(dfs, ignore_index=True)
    pivot = all_df.pivot_table(index="lane", columns="source",
                               values="CollectionTime", aggfunc="count",
                               fill_value=0)
    # Preserve source order based on first appearance
    source_order = []
    for df in dfs:
        lbl = df["source"].iloc[0] if len(df) else None
        if lbl is not None and lbl not in source_order:
            source_order.append(lbl)
    pivot = pivot[source_order]
    pivot.loc["TOTAL"] = pivot.sum(axis=0)
    return pivot


def bin_breakdown(dfs: list[pd.DataFrame], bin_minutes: int) -> pd.DataFrame:
    """Long-format: source, lane, bin_start (floored), count."""
    all_df = pd.concat(dfs, ignore_index=True)
    freq = f"{bin_minutes}min"
    all_df["bin_start"] = all_df["CollectionTime"].dt.floor(freq)
    grp = (all_df.groupby(["source", "lane", "bin_start"])
                 .size().rename("count").reset_index())
    return grp.sort_values(["lane", "bin_start", "source"])


def format_bin_tables(bins: pd.DataFrame, source_order: list[str]) -> str:
    """Pretty per-lane time-bin tables."""
    out = []
    for lane in sorted(bins["lane"].unique()):
        sub = bins[bins["lane"] == lane]
        wide = sub.pivot_table(index="bin_start", columns="source",
                               values="count", fill_value=0)
        wide = wide.reindex(columns=source_order, fill_value=0)
        out.append(f"\n### {lane}")
        out.append(wide.to_string())
    return "\n".join(out)


def main():
    args = parse_args()

    paths = [args.vendor] + list(args.baseline)
    if args.labels:
        labels = [s.strip() for s in args.labels.split(",")]
        if len(labels) != len(paths):
            raise SystemExit(
                f"--labels has {len(labels)} items but there are {len(paths)} sources"
            )
    else:
        labels = [p.stem for p in paths]

    dfs = [load(p, lbl) for p, lbl in zip(paths, labels)]

    # Infer time window from baselines (index 1..) if not explicitly given.
    baseline_dfs = dfs[1:]
    all_baseline_times = pd.concat([d["CollectionTime"] for d in baseline_dfs])
    inferred_start = all_baseline_times.min()
    inferred_end = all_baseline_times.max() + pd.Timedelta(seconds=1)
    start = pd.to_datetime(args.start) if args.start else inferred_start
    end = pd.to_datetime(args.end) if args.end else inferred_end
    dfs = [d[(d["CollectionTime"] >= start) & (d["CollectionTime"] < end)].copy()
           for d in dfs]

    print("=" * 72)
    print(f"Stage-1 evaluation — hour totals by lane")
    print(f"Window: [{start}, {end})  "
          f"{'(inferred from baselines)' if not (args.start and args.end) else '(user)'}")
    print(f"Vendor: {args.vendor.name}")
    for p, lbl in zip(args.baseline, labels[1:]):
        print(f"Baseline: {p.name}  [{lbl}]")
    print("=" * 72)

    totals = hour_totals(dfs)
    # Add deltas vs vendor for each baseline
    vendor_label = labels[0]
    for lbl in labels[1:]:
        totals[f"Δ({lbl}-{vendor_label})"] = totals[lbl] - totals[vendor_label]
    print(totals.to_string())

    print("\n" + "=" * 72)
    print(f"Per-lane {args.bin_minutes}-min bin breakdown")
    print("=" * 72)
    bins = bin_breakdown(dfs, args.bin_minutes)
    print(format_bin_tables(bins, labels))

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        bins.to_csv(args.out, index=False)
        print(f"\n[saved] {args.out}")

    # Short verdict: is v7 closer to vendor than v3 on East_01?
    if len(labels) >= 3 and "East_01" in totals.index:
        v = totals.loc["East_01", vendor_label]
        diffs = {lbl: abs(totals.loc["East_01", lbl] - v) for lbl in labels[1:]}
        best = min(diffs, key=diffs.get)
        print(f"\n[East_01] vendor={v}; closest baseline: {best} (|Δ|={diffs[best]})")


if __name__ == "__main__":
    main()
