"""
TARGET FACTORY
--------------

This module builds a deterministic target grid for a scenario and places one
target at the center of every valid grid cell.

Target rules:
- The target can be anywhere in the environment.
- The target (for hardware pusposes) is a ClearCom belpack
- Cells are generated across the full environment extent.
- A cell is valid if its center is not inside a wall footprint.
- Rooms, corridors, patios, and exterior free space are all valid target areas.

The grid uses a finer maximum spacing indoors and a coarser spacing outdoors:
- indoor: 5 m
- outdoor: 10 m

@author: Giuliana Emberson
@date: 7th of May 2026
"""

from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Iterable, List, Optional, Tuple
from environment_factory import Environment
from floor_plan_factory import EPSILON, FloorElement, GeneratedFloorPlan, Patio, Rect, Room


@dataclass(frozen=True)
class TargetGridCell:
    cell_id: str
    row_idx: int
    col_idx: int
    x_center: float
    y_center: float
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    cell_width: float
    cell_height: float
    is_valid: bool
    space_type: str
    room_id: Optional[str] = None
    room_type: Optional[str] = None
    patio_id: Optional[str] = None


@dataclass(frozen=True)
class Target:
    target_id: int
    target_label: str
    position: Tuple[float, float]
    cell_id: str
    row_idx: int
    col_idx: int
    cell_width: float
    cell_height: float
    space_type: str
    room_id: Optional[str] = None
    room_type: Optional[str] = None
    patio_id: Optional[str] = None


@dataclass(frozen=True)
class GeneratedTargetGrid:
    requested_max_cell_size: float
    rows: int
    cols: int
    cell_width: float
    cell_height: float
    cells: List[TargetGridCell]
    targets: List[Target]


class TargetFactory:
    _INDOOR_MAX_CELL_SIZE_M = 5.0
    _OUTDOOR_MAX_CELL_SIZE_M = 10.0

    def __init__(
        self,
        env: Environment,
        floor_plan: GeneratedFloorPlan,
        *,
        max_cell_size: Optional[float] = None,
    ) -> None:
        self.env = env
        self.floor_plan = floor_plan
        self.max_cell_size = max_cell_size or self._default_max_cell_size()

    def generate(self) -> GeneratedTargetGrid:
        rows, cols, cell_width, cell_height = self._grid_shape()
        wall_rects = self._wall_rectangles()

        cells: List[TargetGridCell] = []
        targets: List[Target] = []
        target_counter = 0

        x0, x1 = self.env.x_domain
        y0, y1 = self.env.y_range

        for row_idx in range(rows):
            for col_idx in range(cols):
                cell_x_min = x0 + (col_idx * cell_width)
                cell_x_max = min(x1, cell_x_min + cell_width)
                cell_y_min = y0 + (row_idx * cell_height)
                cell_y_max = min(y1, cell_y_min + cell_height)
                center = (
                    (cell_x_min + cell_x_max) / 2.0,
                    (cell_y_min + cell_y_max) / 2.0,
                )

                is_valid = not self._point_inside_any_rect(center, wall_rects)
                space_type, room_id, room_type, patio_id = self._classify_point(center, is_valid=is_valid)
                cell_id = self._cell_id(row_idx, col_idx)
                cell = TargetGridCell(
                    cell_id=cell_id,
                    row_idx=row_idx,
                    col_idx=col_idx,
                    x_center=center[0],
                    y_center=center[1],
                    x_min=cell_x_min,
                    y_min=cell_y_min,
                    x_max=cell_x_max,
                    y_max=cell_y_max,
                    cell_width=cell_x_max - cell_x_min,
                    cell_height=cell_y_max - cell_y_min,
                    is_valid=is_valid,
                    space_type=space_type,
                    room_id=room_id,
                    room_type=room_type,
                    patio_id=patio_id,
                )
                cells.append(cell)

                if is_valid:
                    targets.append(
                        Target(
                            target_id=target_counter,
                            target_label=self._target_label(target_counter),
                            position=center,
                            cell_id=cell_id,
                            row_idx=row_idx,
                            col_idx=col_idx,
                            cell_width=cell.cell_width,
                            cell_height=cell.cell_height,
                            space_type=space_type,
                            room_id=room_id,
                            room_type=room_type,
                            patio_id=patio_id,
                        )
                    )
                    target_counter += 1

        return GeneratedTargetGrid(
            requested_max_cell_size=self.max_cell_size,
            rows=rows,
            cols=cols,
            cell_width=cell_width,
            cell_height=cell_height,
            cells=cells,
            targets=targets,
        )

    def _default_max_cell_size(self) -> float:
        if self.env.env_type == "outdoor":
            return self._OUTDOOR_MAX_CELL_SIZE_M
        return self._INDOOR_MAX_CELL_SIZE_M

    def _grid_shape(self) -> Tuple[int, int, float, float]:
        rows = max(1, math.ceil(self.env.height / self.max_cell_size))
        cols = max(1, math.ceil(self.env.width / self.max_cell_size))
        cell_width = self.env.width / cols
        cell_height = self.env.height / rows
        return rows, cols, cell_width, cell_height

    def _cell_id(self, row_idx: int, col_idx: int) -> str:
        return f"cell_r{row_idx:04d}_c{col_idx:04d}"

    def _target_label(self, target_id: int) -> str:
        return f"target_{target_id:05d}"

    def _wall_rectangles(self) -> List[Rect]:
        wall_rects: List[Rect] = []
        for element in self.floor_plan.elements:
            if element.element_type != "wall" or element.length is None or element.thickness is None:
                continue
            wall_rects.append(
                self._rotated_wall_aabb(
                    anchor_x=element.x,
                    anchor_y=element.y,
                    length=element.length,
                    thickness=element.thickness,
                    orientation=element.orientation_angle,
                )
            )
        return wall_rects

    def _rotated_wall_aabb(
        self,
        *,
        anchor_x: float,
        anchor_y: float,
        length: float,
        thickness: float,
        orientation: float,
    ) -> Rect:
        angle = math.radians(orientation % 360.0)
        cos_angle = math.cos(angle)
        sin_angle = math.sin(angle)
        local_corners = (
            (0.0, 0.0),
            (length, 0.0),
            (length, thickness),
            (0.0, thickness),
        )
        world_corners = []
        for x_local, y_local in local_corners:
            world_x = anchor_x + (x_local * cos_angle) - (y_local * sin_angle)
            world_y = anchor_y + (x_local * sin_angle) + (y_local * cos_angle)
            world_corners.append((world_x, world_y))

        xs = [point[0] for point in world_corners]
        ys = [point[1] for point in world_corners]
        return Rect(
            x_min=min(xs) - EPSILON,
            y_min=min(ys) - EPSILON,
            x_max=max(xs) + EPSILON,
            y_max=max(ys) + EPSILON,
        )

    def _point_inside_any_rect(
        self,
        point: Tuple[float, float],
        rects: Iterable[Rect],
    ) -> bool:
        return any(self._point_inside_rect(point, rect) for rect in rects)

    def _point_inside_rect(self, point: Tuple[float, float], rect: Rect) -> bool:
        x, y = point
        return (
            rect.x_min - EPSILON <= x <= rect.x_max + EPSILON
            and rect.y_min - EPSILON <= y <= rect.y_max + EPSILON
        )

    def _classify_point(
        self,
        point: Tuple[float, float],
        *,
        is_valid: bool,
    ) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
        if not is_valid:
            return ("wall", None, None, None)

        for patio in self.floor_plan.patios:
            if self._point_inside_rect(point, patio.rect):
                return ("patio", None, None, patio.patio_id)

        for room in self.floor_plan.rooms:
            if self._point_inside_rect(point, room.rect):
                return ("room", room.room_id, room.room_type, None)

        if self._point_inside_rect(point, self.floor_plan.building_rect):
            return ("building_free", None, None, None)

        return ("exterior", None, None, None)


def generate_targets_from_floor_plan(
    environment: Environment,
    floor_plan: GeneratedFloorPlan,
    *,
    max_cell_size: Optional[float] = None,
) -> GeneratedTargetGrid:
    return TargetFactory(environment, floor_plan, max_cell_size=max_cell_size).generate()
