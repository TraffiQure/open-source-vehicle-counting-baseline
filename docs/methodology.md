# Methodology

This document describes the open-source baseline pipeline in enough detail to understand the results it produces and to reproduce them on new video.

## Purpose

The pipeline is an *independent* vehicle-counting system that produces per-vehicle records (timestamp, direction, lane, FHWA 13-class type) from raw traffic video. It is designed to be run side-by-side with a commercial vendor's report on the same source video, so that the two outputs can be compared row-by-row.

It is **not** a ground-truth source. Hand-counted visual samples (10-minute video segments counted manually into a lane × class grid) are the closest approximation of ground truth and are used as the reference whenever any two systems are compared.

## Pipeline components

### Detector (stage 1)

- **Model**: YOLO11s (Ultralytics) — a small-footprint single-stage detector.
- **Training data**: [MIO-TCD localization](https://tcd.miovision.com/) — ≈110k overhead-view vehicle images covering 11 classes (articulated truck, bicycle, bus, car, motorcycle, motorised non-motor, pedestrian, pickup truck, single-unit truck, work van, background).
- **Fine-tune procedure**: see `src/train_stage1.py`. The released checkpoint (`yolo11s-miotcd-v2-epoch3.pt`) was selected from a longer training run that crashed; epoch 3 produced the best validation mAP before the crash.
- **Inference**: per-frame bounding boxes with class IDs and confidences.

### Tracker

- **Algorithm**: ByteTrack (the variant shipped with Ultralytics).
- **Purpose**: associate detections across frames so each unique vehicle gets a stable track ID. Without tracking the same vehicle would be counted multiple times as it traverses a lane.
- **Configuration**: default ByteTrack hyperparameters.

### Counter

- **Setup**: one virtual line per traffic lane, hand-drawn on the first frame of each video using `src/annotate_lines.py`. Each line has a designated "arrival side" — the side the camera sees vehicles approaching from.
- **Crossing rule**: a tracked vehicle is counted as crossing the line when its bounding-box centre transitions from the arrival side to the opposite side **and** the projection of the centre onto the line falls within the line's segment (with a ±6 pixel tolerance). The on-segment requirement prevents vehicles passing well outside the lane from being miscounted.
- **Lane assignment**: the lane label comes from whichever line the vehicle crosses. There is no separate lane classifier; the lane-drawing position is the lane definition.

### Class mapping

- The MIO-TCD detector class is mapped to the FHWA 13-class scheme via a hard-class routing table (`configs/miotcd_to_fhwa.json`). For example, MIO-TCD `car` → FHWA class 2 (passenger car), `pickup_truck` → FHWA class 3.
- For trucks (MIO-TCD `articulated_truck` or `single_unit_truck`), the cropped vehicle image is additionally passed to a stage-2 axle classifier that refines the FHWA assignment (e.g., 3-axle vs 5-axle articulated). See `src/stage2_classifier.py`.

### Stage-2 axle classifier

- **Model**: EfficientNet backbone with a small classification head.
- **Training data**: hand-labelled truck crops from MIO-TCD and from the evaluated videos.
- **Two separate models**: one for articulated trucks, one for single-unit trucks.
- **Released as**: `axle-articulated-v5_fixed.pt` and `axle-single_unit-v5_fixed.pt`.

## Visual ground truth (hand-counted samples)

For each site we hand-count one or more 10-minute video segments and record per-lane × per-class vehicle counts. These are the closest approximation of ground truth in this report. Hand-counted samples are deliberately short (10 minutes) — long enough to give a meaningful per-class count, short enough to actually count by hand.

Each sample is fully independent of both the vendor's output and the baseline's output — the human counter does not see either system's predictions while counting.

## Per-row matching

When comparing two systems' per-vehicle outputs (e.g., baseline vs Vendor L), the matching rule used throughout this work is:

- **Match window**: ±5 seconds, same direction
- **Match definition**: for each vehicle reported by the first system, find the nearest vehicle of the same direction in the second system within ±5 seconds
- **Class agreement / disagreement** is then computed on the matched pairs

## Known limitations

These are real, reproducible properties of the released checkpoint, not bugs:

| Limitation | Impact | Workaround |
|---|---|---|
| FHWA class 5 (single-unit truck) recall ≈ 0 | The baseline systematically under-reports class 5. Class-5 totals should not be compared 1:1 against vendor outputs. | A re-training pass with class-balanced sampling is planned. |
| FHWA class 1 (motorcycle) recall weak | Small-object detection is the weakest single-class signal. | Comparison against vendor and GT must account for this. |
| Low-light / night recall drop | At night and twilight, recall drops to 30–70% vs daylight. | Use hand-counted GT or a vendor count as reference under low light. |
| Snow false positives | Falling snow is sometimes detected as vehicles, especially at night. | Document snow conditions per window and treat baseline snow counts with caution. |
| Fisheye / wide-angle edge distortion | Lanes near the top or bottom edge of a fisheye frame have lower recall than centre lanes (the centre of optical distortion matters, not absolute distance from camera). | When evaluating a fisheye site, expect edge lanes to under-count. |
| Construction-zone lane shifts | If actual vehicle paths shift away from painted lane positions, but counting lines were drawn against painted lanes, lane assignment becomes unreliable. | Re-draw counting lines on the shifted layout (as was done for Station 83 in the published evaluation). |

## Output format

The pipeline writes `PerVehicle_*.csv` files matching the Vendor L schema:

```
Data File:, /path/to/output.csv,   AI Count Version:, opensource-v1
Station Number:, <station>
Site:, <site name>
Location:, <location>
Report Start Time:, <YYYY/MM/DD  HH:MM:SS>
Base Direction:, <e.g. West>
latitude:, 0
longitude:, 0
Lane number Base direction:, <n>
Lane number second direction:, <n>
Classification Type:,  13 Class Classification
Unit Serial Number:, opensource-v1

CollectionTime, Direction, Lane, Class, Speed, videoFileNum, FrameNum, ImageFileName
2026-03-09 07:00:01, West, 2, 2, 000, 1675, 000014,
2026-03-09 07:00:02, West, 1, 2, 000, 1675, 000023,
...
```

`Speed` is currently always 0 — the baseline does not yet compute per-vehicle speed. All other columns match Vendor L's format exactly so that downstream comparison tools work on either output.

## Repository version

The released configuration is tagged **v0.1** and matches the configuration used in the accompanying evaluation report.
