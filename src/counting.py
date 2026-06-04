"""Line-crossing logic for the baseline vehicle counter."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


ON_SEGMENT_TOL_PX = 6


def side_of_line(
    point: tuple[float, float], p1: tuple[int, int], p2: tuple[int, int]
) -> float:
    return ((p2[0] - p1[0]) * (point[1] - p1[1])) - (
        (p2[1] - p1[1]) * (point[0] - p1[0])
    )


def projects_onto_segment(
    point: tuple[float, float],
    p1: tuple[int, int],
    p2: tuple[int, int],
    tol_px: float = ON_SEGMENT_TOL_PX,
) -> bool:
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    length_sq = dx * dx + dy * dy
    if length_sq == 0:
        return True
    length = length_sq ** 0.5
    t = ((point[0] - p1[0]) * dx + (point[1] - p1[1]) * dy) / length
    return -tol_px <= t <= length + tol_px


def arrival_side_sign(
    p1: tuple[int, int], p2: tuple[int, int], hint: tuple[int, int]
) -> int:
    return 1 if side_of_line(hint, p1, p2) > 0 else -1


@dataclass(frozen=True, slots=True)
class CrossEvent:
    track_id: int
    line_name: str
    direction: str
    lane: int
    frame_idx: int
    timestamp: datetime
    coco_class: int


@dataclass(slots=True)
class _TrackLineState:
    last_sign: int | None = None
    counted: bool = False


@dataclass(slots=True)
class _PreparedLine:
    name: str
    direction: str
    lane: int
    p1: tuple[int, int]
    p2: tuple[int, int]
    arrival_sign: int


class LineCounter:
    def __init__(self, lines: list[dict[str, Any]]) -> None:
        if not lines:
            raise ValueError("At least one counting line is required")
        self._lines = [self._prepare_line(line) for line in lines]
        self._track_state: dict[int, dict[str, _TrackLineState]] = {}

    def update(
        self,
        track_id: int,
        bbox_center_xy: tuple[float, float],
        frame_idx: int,
        timestamp: datetime,
        coco_class: int,
    ) -> list[CrossEvent]:
        events: list[CrossEvent] = []
        track_lines = self._track_state.setdefault(track_id, {})

        for line in self._lines:
            state = track_lines.setdefault(line.name, _TrackLineState())
            current_value = side_of_line(bbox_center_xy, line.p1, line.p2)
            current_sign = _sign(current_value)

            if current_sign == 0:
                continue

            if (
                not state.counted
                and state.last_sign == -line.arrival_sign
                and current_sign == line.arrival_sign
                and projects_onto_segment(bbox_center_xy, line.p1, line.p2)
            ):
                state.counted = True
                events.append(
                    CrossEvent(
                        track_id=track_id,
                        line_name=line.name,
                        direction=line.direction,
                        lane=line.lane,
                        frame_idx=frame_idx,
                        timestamp=timestamp,
                        coco_class=coco_class,
                    )
                )

            state.last_sign = current_sign

        return events

    def _prepare_line(self, line: dict[str, Any]) -> _PreparedLine:
        try:
            p1 = (int(line["p1"][0]), int(line["p1"][1]))
            p2 = (int(line["p2"][0]), int(line["p2"][1]))
            hint = (
                int(line["arrival_side_hint"][0]),
                int(line["arrival_side_hint"][1]),
            )
            return _PreparedLine(
                name=str(line["name"]),
                direction=str(line["direction"]),
                lane=int(line["lane"]),
                p1=p1,
                p2=p2,
                arrival_sign=arrival_side_sign(p1, p2, hint),
            )
        except KeyError as exc:
            raise ValueError(f"Invalid line definition missing key: {exc}") from exc


def _sign(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0
