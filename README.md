# open-source-vehicle-counting-baseline

An independent, open-source vehicle-counting pipeline for evaluating commercial traffic-camera vendors. Produces per-vehicle records (timestamp, direction, lane, FHWA 13-class type) from overhead-view traffic video, designed to be run side-by-side with vendor outputs so that the two can be compared row-by-row.

Originally developed by [TraffiQure Technologies](https://www.traffiqure.com) for a vendor-evaluation study for the District Department of Transportation (DDOT) and the Metropolitan Washington Council of Governments (MWCOG).

---

## What this pipeline does

For each traffic video clip, the pipeline:

1. **Detects** vehicles in every frame with a YOLO11s detector fine-tuned on the [MIO-TCD localization dataset](https://tcd.miovision.com/) (≈110k overhead-view vehicle images, 11 classes).
2. **Tracks** vehicles across frames using [ByteTrack](https://arxiv.org/abs/2110.06864) to maintain a stable per-vehicle ID.
3. **Counts** a vehicle as crossing a lane when its bounding-box centre crosses one of the user-drawn per-lane virtual lines, projected within ±6 px of the line segment.
4. **Classifies** trucks into FHWA 13-class buckets (articulated 5-axle vs. single-unit, etc.) with a stage-2 EfficientNet axle classifier trained on hand-labelled MIO-TCD truck crops.
5. **Writes** a `PerVehicle_*.csv` file matching the Vendor L output schema, so the two reports can be diffed directly.

The pipeline is independent of the vendor's reports and uses only the raw video as input.

## Repository contents

```
src/                  pipeline source code
  run_baseline.py     entry point: video → PerVehicle CSV
  annotate_lines.py   GUI to draw per-lane counting lines on a video frame
  counting.py         line-crossing logic (on-segment projection check)
  visualize_crossings.py  overlay counting lines + crossing events on a clip
  vendor_loader.py    parser for Vendor L PerVehicle / BinnedClass / BinnedVolume CSVs
  writer.py           writer for PerVehicle CSV output
  stage2_classifier.py    inference wrapper for the stage-2 axle classifier
  prepare_miotcd.py   convert MIO-TCD into YOLO training format
  train_stage1.py     fine-tune YOLO11s on MIO-TCD localization
  train_stage2_axle.py    train the stage-2 axle classifier
  eval_stage1.py      evaluate stage-1 detector on a held-out split

scripts/              standalone analysis / review tools
  render_annotated_clip.py    render an MP4 with detection boxes + counting-line flashes
  dump_mismatch_frames.py     shared helpers for per-row dump scripts
  dump_ours_only_frames.py    dump frames where baseline reports a vehicle but vendor doesn't
  dump_class_disagree_frames.py   dump frames where baseline and vendor disagree on class
  dump_vendor_class_frames.py     dump frames where vendor reports a class the baseline misses
  review_mismatch_gui.py      Tk GUI to label dumped strips as correct / wrong / ambiguous
  compare_to_vendor.py        per-window summary table (baseline vs vendor)
  draw_lines_on_frame.py      render an annotated still showing the configured counting lines
  verify_framenum.py          sanity-check timestamp ↔ FrameNum alignment

configs/              configuration files
  miotcd.yaml             YOLO dataset descriptor for MIO-TCD localization
  miotcd_to_fhwa.json     MIO-TCD class → FHWA 13-class routing table
  lines_*.json            example line configs for the evaluated stations (drawn with annotate_lines)

docs/
  methodology.md      full pipeline methodology (extracted from the evaluation report)
```

## Installation

```bash
git clone https://github.com/TraffiQure/open-source-vehicle-counting-baseline.git
cd open-source-vehicle-counting-baseline
conda env create -f environment.yml
conda activate vendorl-eval
```

The environment requires Python 3.11. CPU and Apple Silicon (MPS) are supported out of the box; CUDA users may need to install a matching torch wheel.

## Pre-trained model weights

Pre-trained weights are released as GitHub Release attachments rather than committed to git. From the **v0.1-phase1-station85** release (https://github.com/TraffiQure/open-source-vehicle-counting-baseline/releases/tag/v0.1-phase1-station85), download:

- `yolo11s-miotcd-v2-epoch3.pt` — stage-1 detector (≈54 MB)
- `axle-articulated-v5_fixed.pt` — stage-2 articulated-truck axle classifier (≈16 MB)
- `axle-single_unit-v5_fixed.pt` — stage-2 single-unit-truck axle classifier (≈16 MB)

Place them under `weights/` (or anywhere; `--model` and stage-2 paths take any file path).

## Quick start: run the baseline on one video

```bash
# 1. Draw counting lines on the first frame of the video (interactive)
python -m src.annotate_lines \
  --video path/to/your_video.avi \
  --out configs/lines_my_site.json

# 2. Run the pipeline → PerVehicle CSV
python -m src.run_baseline \
  --video path/to/your_video.avi \
  --lines configs/lines_my_site.json \
  --model weights/yolo11s-miotcd-v2-epoch3.pt \
  --class-map configs/miotcd_to_fhwa.json \
  --out outputs/PerVehicle_my_site.csv
```

The output CSV matches the Vendor L PerVehicle schema (`CollectionTime, Direction, Lane, Class, Speed, videoFileNum, FrameNum, ImageFileName`), so it can be diffed directly against a vendor file.

## Reproducing the DDOT / MWCOG evaluation

The line-config files used in the published evaluation are checked into `configs/`:

- `lines_85_20260308070018__1651.json` — Station 85 (Benning Rd EB, Arterial)
- `lines_81_20260224065418__2656.json` — Station 81 (Freeway, southbound 4-lane)
- `lines_82_20260304070015__1677.json` — Station 82 (Freeway, bidirectional)
- `lines_83_20260304070017__2634.json` — Station 83 (Freeway, construction zone)

To reproduce per-station numbers, run `src.run_baseline` on each video clip with the corresponding line config and stage-1 model weights from the v0.1 release. The full report PDF / DOCX is published separately.

## Known limitations

These are the same limitations called out in the evaluation report's *Methodology* section:

- **FHWA class 5 (single-unit trucks)** — current recall is essentially 0. The stage-1 detector does not route trucks to the single-unit branch reliably; a re-training pass is planned.
- **FHWA class 1 (motorcycles)** — small-object recall is weak.
- **Low-light / night** — detector recall drops sharply (often 30–70%) at night or in twilight. The model is not specifically trained on night-time imagery.
- **Snowfall** — falling snow can be detected as vehicles (false positives).
- **Fisheye / wide-angle lenses** — lanes at the top and bottom edges of the frame are more distorted and have lower recall than centre lanes (see Station 81 in the report).

## Training

To re-train stage 1 on MIO-TCD (≈4–8 h on a modern GPU):

```bash
python -m src.prepare_miotcd --src path/to/MIO-TCD --out data/miotcd
python -m src.train_stage1 --data data/miotcd --epochs 30 --imgsz 640
```

To re-train a stage-2 axle classifier (articulated or single-unit):

```bash
python -m src.train_stage2_axle --class-name articulated --data data/axle_crops
```

## License

[MIT](LICENSE) — © 2026 TraffiQure Technologies.

## Citation

If you use this software in academic work, please cite the accompanying evaluation report:

> TraffiQure Technologies (2026). *Vendor L vs open-source baseline — multi-site evaluation* (DDOT / MWCOG vendor-evaluation report).

## Acknowledgements

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) — detection backbone
- [ByteTrack](https://github.com/ifzhang/ByteTrack) — tracking
- [MIO-TCD](https://tcd.miovision.com/) — overhead-view training data
- District Department of Transportation (DDOT) and Metropolitan Washington Council of Governments (MWCOG) — providing the source video and per-vehicle vendor outputs that motivated this work.
