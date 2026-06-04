"""OpenCV line annotation tool for per-lane counting lines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2


class LineAnnotator:
    def __init__(self, video_path: Path, lines_path: Path | None, out_path: Path) -> None:
        self.video_path = video_path
        self.lines_path = lines_path
        self.out_path = out_path
        self.window_name = f"annotate_lines — {video_path.name}"
        self.capture = cv2.VideoCapture(str(video_path))
        if not self.capture.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        self.lines_data: dict[str, Any] = {"lines": []}
        self.current_frame_index = 0
        self.current_frame = self._read_reference_frame()
        self.pending_points: list[tuple[int, int]] = []
        self.unsaved_changes = False

        if lines_path is not None and lines_path.exists():
            self.lines_data = json.loads(lines_path.read_text(encoding="utf-8"))
            self._validate_lines_data(self.lines_data)
            self.current_frame_index = int(self.lines_data["reference_frame_index"])
            self.current_frame = self._read_reference_frame()

    def run(self) -> None:
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._on_mouse)
        self._redraw()

        while True:
            key = cv2.waitKey(20) & 0xFF
            if key == 255:
                continue
            if key == ord("s"):
                self._save()
            elif key == ord("d"):
                self._delete_last_line()
            elif key == ord("c"):
                self._clear_lines()
            elif key == ord("f"):
                self._advance_frame()
            elif key == ord("q"):
                if self._confirm_quit():
                    break

        self.capture.release()
        cv2.destroyAllWindows()

    def _on_mouse(self, event: int, x: int, y: int, _flags: int, _param: Any) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        self.pending_points.append((int(x), int(y)))
        self._redraw()
        if len(self.pending_points) == 3:
            self._finalize_line()

    def _read_reference_frame(self) -> Any:
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, float(self.current_frame_index))
        ok, frame = self.capture.read()
        if not ok or frame is None:
            raise ValueError(
                f"Could not read frame {self.current_frame_index} from {self.video_path}"
            )
        return frame

    def _finalize_line(self) -> None:
        p1, p2, hint = self.pending_points
        direction = self._prompt_direction()
        lane = self._prompt_lane()
        self.lines_data["lines"].append(
            {
                "name": f"{direction}_{lane:02d}",
                "direction": direction,
                "lane": lane,
                "p1": [p1[0], p1[1]],
                "p2": [p2[0], p2[1]],
                "arrival_side_hint": [hint[0], hint[1]],
            }
        )
        self.pending_points = []
        self.unsaved_changes = True
        self._redraw()

    def _prompt_direction(self) -> str:
        valid_directions = {"East", "West", "North", "South"}
        while True:
            value = input("Direction (East/West/North/South): ").strip().title()
            if value in valid_directions:
                return value
            print("Invalid direction.")

    def _prompt_lane(self) -> int:
        while True:
            value = input("Lane (integer >= 1): ").strip()
            try:
                lane = int(value)
            except ValueError:
                print("Lane must be an integer.")
                continue
            if lane >= 1:
                return lane
            print("Lane must be >= 1.")

    def _delete_last_line(self) -> None:
        if self.lines_data["lines"]:
            self.lines_data["lines"].pop()
            self.pending_points = []
            self.unsaved_changes = True
            self._redraw()

    def _clear_lines(self) -> None:
        if self.lines_data["lines"]:
            self.lines_data["lines"] = []
            self.pending_points = []
            self.unsaved_changes = True
            self._redraw()

    def _advance_frame(self) -> None:
        self.current_frame_index += 1
        self.current_frame = self._read_reference_frame()
        self.pending_points = []
        self._redraw()

    def _save(self) -> None:
        frame_height, frame_width = self.current_frame.shape[:2]
        payload = {
            "video": self.video_path.name,
            "frame_size": [frame_width, frame_height],
            "reference_frame_index": self.current_frame_index,
            "lines": self.lines_data["lines"],
        }
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.lines_data = payload
        self.unsaved_changes = False
        print(f"Saved {self.out_path}")

    def _confirm_quit(self) -> bool:
        if not self.unsaved_changes:
            return True
        response = input("Unsaved changes exist. Quit without saving? [y/N]: ").strip()
        return response.lower() == "y"

    def _redraw(self) -> None:
        canvas = self.current_frame.copy()
        for line in self.lines_data["lines"]:
            self._draw_line(canvas, line)
        self._draw_pending(canvas)
        cv2.imshow(self.window_name, canvas)

    def _draw_line(self, canvas: Any, line: dict[str, Any]) -> None:
        p1 = tuple(int(value) for value in line["p1"])
        p2 = tuple(int(value) for value in line["p2"])
        hint = tuple(int(value) for value in line["arrival_side_hint"])
        midpoint = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)

        cv2.line(canvas, p1, p2, (0, 255, 0), 2)
        cv2.arrowedLine(canvas, midpoint, hint, (0, 0, 255), 2, tipLength=0.15)
        cv2.circle(canvas, p1, 4, (255, 255, 0), -1)
        cv2.circle(canvas, p2, 4, (255, 255, 0), -1)
        cv2.putText(
            canvas,
            str(line["name"]),
            (midpoint[0] + 6, midpoint[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def _draw_pending(self, canvas: Any) -> None:
        if not self.pending_points:
            return
        cv2.circle(canvas, self.pending_points[0], 4, (0, 255, 255), -1)
        if len(self.pending_points) >= 2:
            cv2.circle(canvas, self.pending_points[1], 4, (0, 255, 255), -1)
            cv2.line(canvas, self.pending_points[0], self.pending_points[1], (0, 255, 255), 2)
        if len(self.pending_points) == 3:
            midpoint = (
                (self.pending_points[0][0] + self.pending_points[1][0]) // 2,
                (self.pending_points[0][1] + self.pending_points[1][1]) // 2,
            )
            cv2.arrowedLine(
                canvas,
                midpoint,
                self.pending_points[2],
                (0, 255, 255),
                2,
                tipLength=0.15,
            )

    def _validate_lines_data(self, payload: dict[str, Any]) -> None:
        required_keys = {"video", "frame_size", "reference_frame_index", "lines"}
        missing = required_keys - payload.keys()
        if missing:
            raise ValueError(f"Invalid lines JSON missing keys: {sorted(missing)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--lines", type=Path)
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.video.exists():
        raise FileNotFoundError(args.video)

    out_path = args.out
    if out_path is None:
        out_path = Path("configs") / f"lines_{args.video.stem}.json"

    annotator = LineAnnotator(args.video, args.lines, out_path)
    annotator.run()


if __name__ == "__main__":
    main()
