from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

try:
    from PIL import Image, ImageTk
except ImportError:  # pragma: no cover
    Image = None
    ImageTk = None


RESAMPLE = getattr(getattr(Image, "Resampling", Image), "LANCZOS", None) if Image is not None else None
OUT_FIELDS = [
    "case_idx",
    "vendor_time",
    "vendor_lane",
    "vendor_class",
    "ours_nearest_time",
    "ours_nearest_lane",
    "ours_nearest_delta_s",
    "verdict",
    "notes",
    "reviewed_at",
]


@dataclass(frozen=True, slots=True)
class MismatchRecord:
    case_idx: str
    vendor_time: str
    vendor_lane: str
    vendor_class: str
    ours_nearest_time: str
    ours_nearest_lane: str
    ours_nearest_delta_s: str
    strip_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strip-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_reviewed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["case_idx"] for row in csv.DictReader(handle) if row.get("case_idx")}


def append_review(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUT_FIELDS, lineterminator="\n")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def load_records(strip_dir: Path, out_path: Path, resume: bool) -> tuple[list[MismatchRecord], int]:
    index_path = strip_dir / "index.csv"
    if not index_path.exists():
        raise SystemExit(f"Missing index.csv in {strip_dir}")

    reviewed = load_reviewed(out_path) if resume else set()
    records: list[MismatchRecord] = []
    skipped_missing = 0

    with index_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            case_idx = (row.get("case_idx") or "").strip()
            if not case_idx or case_idx in reviewed:
                continue
            raw_strip = (row.get("strip_path") or "").strip()
            strip_path = Path(raw_strip)
            if not strip_path.is_absolute():
                strip_path = strip_dir / strip_path
            if not strip_path.exists():
                skipped_missing += 1
                print(f"warning: missing strip PNG, skipping: {strip_path}", file=sys.stderr)
                continue
            records.append(
                MismatchRecord(
                    case_idx=case_idx,
                    vendor_time=(row.get("vendor_time") or "").strip(),
                    vendor_lane=(row.get("vendor_lane") or "").strip(),
                    vendor_class=(row.get("vendor_class") or "").strip(),
                    ours_nearest_time=(row.get("ours_nearest_west_time") or "").strip(),
                    ours_nearest_lane=(row.get("ours_nearest_west_lane") or "").strip(),
                    ours_nearest_delta_s=(row.get("ours_nearest_west_delta_s") or "").strip(),
                    strip_path=strip_path,
                )
            )
    return records, len(reviewed) + len(records)


class ReviewApp:
    def __init__(self, root: tk.Tk, records: list[MismatchRecord], out_path: Path, reviewed_done: int, total_all: int):
        self.root = root
        self.records = records
        self.out_path = out_path
        self.reviewed_done = reviewed_done
        self.total_all = total_all
        self.index = 0
        self.photo = None
        self.base_image = None
        self.zoom_scale = 1.0
        self.image_id = None

        root.title("Vendor Mismatch Review")
        root.geometry("1100x500")
        root.protocol("WM_DELETE_WINDOW", self.quit_app)

        self.header = ttk.Label(root, font=("TkDefaultFont", 12, "bold"))
        self.header.pack(fill="x", padx=12, pady=(12, 8))
        self.image_canvas = tk.Canvas(root, background="#111111", highlightthickness=0)
        self.image_canvas.pack(fill="both", expand=True, padx=12, pady=8)
        self.image_canvas.bind("<ButtonPress-1>", self.start_pan)
        self.image_canvas.bind("<B1-Motion>", self.pan_image)
        self.meta_vendor = ttk.Label(root, justify="left")
        self.meta_vendor.pack(fill="x", padx=12)
        self.meta_ours = ttk.Label(root, justify="left")
        self.meta_ours.pack(fill="x", padx=12, pady=(0, 8))

        buttons = ttk.Frame(root)
        buttons.pack(fill="x", padx=12, pady=8)
        ttk.Button(buttons, text="Phantom (1)", command=self.mark_phantom).pack(side="left", padx=4)
        ttk.Button(buttons, text="We missed (2)", command=self.mark_missed).pack(side="left", padx=4)
        ttk.Button(buttons, text="Ambiguous (3)", command=self.mark_ambiguous).pack(side="left", padx=4)
        ttk.Button(buttons, text="Zoom +", command=self.zoom_in).pack(side="left", padx=(16, 4))
        ttk.Button(buttons, text="Zoom -", command=self.zoom_out).pack(side="left", padx=4)
        ttk.Button(buttons, text="Fit (0)", command=self.reset_zoom).pack(side="left", padx=4)

        notes = ttk.Frame(root)
        notes.pack(fill="x", padx=12, pady=8)
        ttk.Label(notes, text="Notes:").pack(side="left")
        self.notes_var = tk.StringVar()
        ttk.Entry(notes, textvariable=self.notes_var).pack(side="left", fill="x", expand=True, padx=(8, 0))

        nav = ttk.Frame(root)
        nav.pack(fill="x", padx=12, pady=(8, 12))
        ttk.Button(nav, text="Prev", command=self.prev_record).pack(side="left")
        ttk.Button(nav, text="Next", command=self.next_record).pack(side="right")

        for key, fn in {
            "1": self.mark_phantom,
            "2": self.mark_missed,
            "3": self.mark_ambiguous,
            "<Left>": self.prev_record,
            "<Right>": self.next_record,
            "q": self.quit_app,
            "<Escape>": self.quit_app,
            "+": self.zoom_in,
            "=": self.zoom_in,
            "-": self.zoom_out,
            "0": self.reset_zoom,
        }.items():
            root.bind(key, fn)
        root.bind("<Command-MouseWheel>", self.zoom_mousewheel)
        root.bind("<Control-MouseWheel>", self.zoom_mousewheel)

        self.render()

    def current(self) -> MismatchRecord:
        return self.records[self.index]

    def render(self) -> None:
        record = self.current()
        self.notes_var.set("")
        self.zoom_scale = 1.0
        self.header.config(text=f"[{self.index + 1} / {len(self.records)}]   case_idx={record.case_idx}")
        self.meta_vendor.config(
            text=f"Vendor: {record.vendor_time}  lane={record.vendor_lane}  class={record.vendor_class}"
        )
        ours_time = record.ours_nearest_time or "(none)"
        ours_lane = record.ours_nearest_lane or "-"
        ours_delta = f"{float(record.ours_nearest_delta_s):+.1f}s" if record.ours_nearest_delta_s else ""
        self.meta_ours.config(text=f"Ours nearest: {ours_time}  lane={ours_lane}  D={ours_delta}")
        self.show_image(record.strip_path)

    def show_image(self, path: Path) -> None:
        if Image is None or ImageTk is None:  # pragma: no cover
            self.photo = tk.PhotoImage(file=str(path))
            self.image_canvas.delete("all")
            self.image_id = self.image_canvas.create_image(0, 0, anchor="nw", image=self.photo)
            self.reset_canvas_view()
        else:
            with Image.open(path) as image:
                self.base_image = self.fit_image(image.convert("RGB"))
            self.refresh_image()

    def fit_image(self, image: Image.Image) -> Image.Image:
        fitted = image.copy()
        fitted.thumbnail((1080, 320), RESAMPLE)
        return fitted

    def reset_canvas_view(self) -> None:
        self.image_canvas.update_idletasks()
        self.image_canvas.xview_moveto(0.0)
        self.image_canvas.yview_moveto(0.0)

    def refresh_image(self) -> None:
        if self.base_image is None or ImageTk is None:
            return
        width = max(1, int(round(self.base_image.width * self.zoom_scale)))
        height = max(1, int(round(self.base_image.height * self.zoom_scale)))
        image = self.base_image.resize((width, height), RESAMPLE)
        self.photo = ImageTk.PhotoImage(image)
        self.image_canvas.delete("all")
        self.image_id = self.image_canvas.create_image(0, 0, anchor="nw", image=self.photo)
        self.image_canvas.config(scrollregion=(0, 0, width, height))
        self.reset_canvas_view()

    def zoom_in(self, *_args) -> None:
        if self.base_image is None:
            return
        self.zoom_scale = min(self.zoom_scale * 1.25, 8.0)
        self.refresh_image()

    def zoom_out(self, *_args) -> None:
        if self.base_image is None:
            return
        self.zoom_scale = max(self.zoom_scale / 1.25, 0.25)
        self.refresh_image()

    def reset_zoom(self, *_args) -> None:
        if self.base_image is None:
            return
        self.zoom_scale = 1.0
        self.refresh_image()

    def zoom_mousewheel(self, event: tk.Event) -> None:
        if getattr(event, "delta", 0) > 0:
            self.zoom_in()
        elif getattr(event, "delta", 0) < 0:
            self.zoom_out()

    def start_pan(self, event: tk.Event) -> None:
        self.image_canvas.scan_mark(event.x, event.y)

    def pan_image(self, event: tk.Event) -> None:
        self.image_canvas.scan_dragto(event.x, event.y, gain=1)

    def review_row(self, verdict: str) -> dict[str, str]:
        record = self.current()
        return {
            "case_idx": record.case_idx,
            "vendor_time": record.vendor_time,
            "vendor_lane": record.vendor_lane,
            "vendor_class": record.vendor_class,
            "ours_nearest_time": record.ours_nearest_time,
            "ours_nearest_lane": record.ours_nearest_lane,
            "ours_nearest_delta_s": record.ours_nearest_delta_s,
            "verdict": verdict,
            "notes": self.notes_var.get().strip(),
            "reviewed_at": datetime.now().isoformat(timespec="seconds"),
        }

    def save_and_advance(self, verdict: str) -> None:
        append_review(self.out_path, self.review_row(verdict))
        self.reviewed_done += 1
        if self.index == len(self.records) - 1:
            messagebox.showinfo("Done", f"Done - {self.reviewed_done}/{self.total_all} reviewed")
            self.root.destroy()
            return
        self.index += 1
        self.render()

    def mark_phantom(self, *_args) -> None:
        self.save_and_advance("phantom")

    def mark_missed(self, *_args) -> None:
        self.save_and_advance("missed")

    def mark_ambiguous(self, *_args) -> None:
        self.save_and_advance("ambiguous")

    def prev_record(self, *_args) -> None:
        if self.index > 0:
            self.index -= 1
            self.render()

    def next_record(self, *_args) -> None:
        if self.index < len(self.records) - 1:
            self.index += 1
            self.render()

    def quit_app(self, *_args) -> None:
        self.root.destroy()


def main() -> None:
    args = parse_args()
    if not args.strip_dir.exists():
        raise FileNotFoundError(args.strip_dir)

    records, total_all = load_records(args.strip_dir, args.out, args.resume)
    reviewed_done = len(load_reviewed(args.out)) if args.resume else 0
    if not records:
        raise SystemExit(
            f"All mismatch strips already reviewed in {args.out}. Re-run without --resume to redo."
            if args.resume
            else f"No reviewable mismatch strips found in {args.strip_dir}"
        )

    root = tk.Tk()
    ReviewApp(root, records, args.out, reviewed_done=reviewed_done, total_all=total_all)
    root.mainloop()


if __name__ == "__main__":
    main()
