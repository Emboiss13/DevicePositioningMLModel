"""
LINK FACTORY
------------

This module builds the shared per-link geometry layer used by all positioning
methods. A link is defined between one antenna and one valid target position.

The goal of this file is to keep only the general link information here:
- Euclidean distance
- LOS / NLOS classification
- Obstacle counts
- Obstacle metadata and coordinates

Method-specific measurements such as RSSI, TDOA, or AOA should be derived from
these link rows in their own modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from antenna_factory import Antenna
from floor_plan_factory import EPSILON, GeneratedFloorPlan
from human_factory import Human
from target_factory import Target


Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]


class LinkState(str, Enum):
    LOS = "LOS"
    NLOS = "NLOS"


@dataclass(frozen=True)
class LinkObstacle:
    obstacle_id: str
    obstacle_type: str
    geometry_type: str
    coordinates: Tuple[Point, ...]
    radius: Optional[float] = None

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "obstacle_id": self.obstacle_id,
            "obstacle_type": self.obstacle_type,
            "geometry_type": self.geometry_type,
            "coordinates": [[x, y] for x, y in self.coordinates],
        }
        if self.radius is not None:
            payload["radius"] = self.radius
        return payload


@dataclass(frozen=True)
class Link:
    scenario_id: Optional[str]
    antenna_id: int
    antenna_label: str
    antenna_position: Point
    target_id: int
    target_label: str
    target_position: Point
    target_cell_id: str
    target_row_idx: int
    target_col_idx: int
    target_space_type: str
    target_room_id: Optional[str]
    target_room_type: Optional[str]
    target_patio_id: Optional[str]
    distance_m: float
    link_state: LinkState
    wall_blocker_count: int
    human_blocker_count: int
    total_blocker_count: int
    blocking_obstacles: Tuple[LinkObstacle, ...]

    @property
    def is_los(self) -> bool:
        return self.link_state == LinkState.LOS

    def to_row(self) -> Dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "antenna_id": self.antenna_id,
            "antenna_label": self.antenna_label,
            "antenna_x": self.antenna_position[0],
            "antenna_y": self.antenna_position[1],
            "target_id": self.target_id,
            "target_label": self.target_label,
            "target_x": self.target_position[0],
            "target_y": self.target_position[1],
            "target_cell_id": self.target_cell_id,
            "target_row_idx": self.target_row_idx,
            "target_col_idx": self.target_col_idx,
            "target_space_type": self.target_space_type,
            "target_room_id": self.target_room_id,
            "target_room_type": self.target_room_type,
            "target_patio_id": self.target_patio_id,
            "distance_m": self.distance_m,
            "is_los": self.is_los,
            "link_state": self.link_state.value,
            "wall_blocker_count": self.wall_blocker_count,
            "human_blocker_count": self.human_blocker_count,
            "total_blocker_count": self.total_blocker_count,
            "blocking_obstacles_json": json.dumps(
                [obstacle.to_dict() for obstacle in self.blocking_obstacles],
                separators=(",", ":"),
            ),
        }


@dataclass(frozen=True)
class _ObstacleGeometry:
    obstacle_id: str
    obstacle_type: str
    geometry_type: str
    coordinates: Tuple[Point, ...]
    bbox: BBox
    radius: Optional[float] = None

    def to_public(self) -> LinkObstacle:
        return LinkObstacle(
            obstacle_id=self.obstacle_id,
            obstacle_type=self.obstacle_type,
            geometry_type=self.geometry_type,
            coordinates=self.coordinates,
            radius=self.radius,
        )


def euclidean_distance(p1: Point, p2: Point) -> float:
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return math.hypot(dx, dy)


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def _dot(a: Point, b: Point) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _sub(a: Point, b: Point) -> Point:
    return (a[0] - b[0], a[1] - b[1])


def _segment_bbox(a: Point, b: Point) -> BBox:
    return (
        min(a[0], b[0]),
        min(a[1], b[1]),
        max(a[0], b[0]),
        max(a[1], b[1]),
    )


def _bbox_intersects(a: BBox, b: BBox) -> bool:
    return not (
        a[2] < b[0] - EPSILON
        or b[2] < a[0] - EPSILON
        or a[3] < b[1] - EPSILON
        or b[3] < a[1] - EPSILON
    )


def _orientation(a: Point, b: Point, c: Point) -> float:
    return ((b[0] - a[0]) * (c[1] - a[1])) - ((b[1] - a[1]) * (c[0] - a[0]))


def _point_on_segment(point: Point, start: Point, end: Point) -> bool:
    return (
        min(start[0], end[0]) - EPSILON <= point[0] <= max(start[0], end[0]) + EPSILON
        and min(start[1], end[1]) - EPSILON <= point[1] <= max(start[1], end[1]) + EPSILON
        and abs(_orientation(start, end, point)) <= EPSILON
    )


def _segments_intersect(a1: Point, a2: Point, b1: Point, b2: Point) -> bool:
    o1 = _orientation(a1, a2, b1)
    o2 = _orientation(a1, a2, b2)
    o3 = _orientation(b1, b2, a1)
    o4 = _orientation(b1, b2, a2)

    if (o1 > EPSILON and o2 < -EPSILON or o1 < -EPSILON and o2 > EPSILON) and (
        o3 > EPSILON and o4 < -EPSILON or o3 < -EPSILON and o4 > EPSILON
    ):
        return True

    return (
        _point_on_segment(b1, a1, a2)
        or _point_on_segment(b2, a1, a2)
        or _point_on_segment(a1, b1, b2)
        or _point_on_segment(a2, b1, b2)
    )


def _point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    inside = False
    x, y = point
    j = len(polygon) - 1

    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        intersects = ((yi > y) != (yj > y)) and (
            x < ((xj - xi) * (y - yi) / ((yj - yi) + EPSILON)) + xi
        )
        if intersects:
            inside = not inside
        j = i

    return inside


def _segment_intersects_polygon(a: Point, b: Point, polygon: Sequence[Point]) -> bool:
    if _point_in_polygon(a, polygon) or _point_in_polygon(b, polygon):
        return True

    for idx in range(len(polygon)):
        start = polygon[idx]
        end = polygon[(idx + 1) % len(polygon)]
        if _segments_intersect(a, b, start, end):
            return True

    return False


def _segment_intersects_circle(a: Point, b: Point, center: Point, radius: float) -> bool:
    ab = _sub(b, a)
    ac = _sub(center, a)
    ab_len_sq = _dot(ab, ab)

    if ab_len_sq <= EPSILON:
        return euclidean_distance(a, center) <= radius + EPSILON

    t = clamp(_dot(ac, ab) / ab_len_sq, 0.0, 1.0)
    closest = (a[0] + ab[0] * t, a[1] + ab[1] * t)
    return euclidean_distance(closest, center) <= radius + EPSILON


def _bbox_from_points(points: Iterable[Point]) -> BBox:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return (min(xs), min(ys), max(xs), max(ys))


def _rotated_rect_corners(
    *,
    anchor_x: float,
    anchor_y: float,
    length: float,
    thickness: float,
    orientation: float,
) -> Tuple[Point, Point, Point, Point]:
    angle = math.radians(orientation % 360.0)
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    local_corners = (
        (0.0, 0.0),
        (length, 0.0),
        (length, thickness),
        (0.0, thickness),
    )

    world_corners: List[Point] = []
    for x_local, y_local in local_corners:
        world_x = anchor_x + (x_local * cos_angle) - (y_local * sin_angle)
        world_y = anchor_y + (x_local * sin_angle) + (y_local * cos_angle)
        world_corners.append((world_x, world_y))

    return tuple(world_corners)  # type: ignore[return-value]


def _human_rect_corners(human: Human) -> Tuple[Point, Point, Point, Point]:
    if human.position_X1_Y1 is None:
        raise ValueError("Rectangular human requires position_X1_Y1.")

    x0, y0 = human.position_X_Y
    x1, y1 = human.position_X1_Y1
    min_x, max_x = sorted((x0, x1))
    min_y, max_y = sorted((y0, y1))
    return (
        (min_x, min_y),
        (max_x, min_y),
        (max_x, max_y),
        (min_x, max_y),
    )


class LinkFactory:
    """
    Build one shared geometry row per antenna-target pair.

    The resulting links are the correct baseline for downstream signal models.
    RSSI/TDOA/AOA should add their own measurement columns on top of these links
    instead of rebuilding distance and obstruction logic independently.
    """

    def __init__(
        self,
        antennas: Sequence[Antenna],
        targets: Sequence[Target],
        *,
        floor_plan: Optional[GeneratedFloorPlan] = None,
        humans: Optional[Sequence[Human]] = None,
        scenario_id: Optional[str] = None,
    ) -> None:
        self.antennas = list(antennas)
        self.targets = list(targets)
        self.floor_plan = floor_plan
        self.humans = list(humans or [])
        self.scenario_id = scenario_id
        self._wall_geometries = self._build_wall_geometries()
        self._human_geometries = self._build_human_geometries()

    def build_links(self) -> List[Link]:
        links: List[Link] = []
        for target in self.targets:
            for antenna in self.antennas:
                links.append(self._build_link(antenna, target))
        return links

    def build_rows(self) -> List[Dict[str, object]]:
        return [link.to_row() for link in self.build_links()]

    def _build_link(self, antenna: Antenna, target: Target) -> Link:
        a = antenna.position
        b = target.position
        segment_bbox = _segment_bbox(a, b)

        wall_blockers = self._blocking_obstacles(segment_bbox, a, b, self._wall_geometries)
        human_blockers = self._blocking_obstacles(segment_bbox, a, b, self._human_geometries)
        blocking_obstacles = tuple([*wall_blockers, *human_blockers])
        link_state = LinkState.LOS if not blocking_obstacles else LinkState.NLOS

        return Link(
            scenario_id=self.scenario_id,
            antenna_id=antenna.antenna_id,
            antenna_label=antenna.antenna_label,
            antenna_position=antenna.position,
            target_id=target.target_id,
            target_label=target.target_label,
            target_position=target.position,
            target_cell_id=target.cell_id,
            target_row_idx=target.row_idx,
            target_col_idx=target.col_idx,
            target_space_type=target.space_type,
            target_room_id=target.room_id,
            target_room_type=target.room_type,
            target_patio_id=target.patio_id,
            distance_m=euclidean_distance(a, b),
            link_state=link_state,
            wall_blocker_count=len(wall_blockers),
            human_blocker_count=len(human_blockers),
            total_blocker_count=len(blocking_obstacles),
            blocking_obstacles=blocking_obstacles,
        )

    def _blocking_obstacles(
        self,
        segment_bbox: BBox,
        start: Point,
        end: Point,
        obstacles: Sequence[_ObstacleGeometry],
    ) -> List[LinkObstacle]:
        blockers: List[LinkObstacle] = []
        for obstacle in obstacles:
            if not _bbox_intersects(segment_bbox, obstacle.bbox):
                continue
            if self._obstacle_blocks_segment(obstacle, start, end):
                blockers.append(obstacle.to_public())
        return blockers

    def _obstacle_blocks_segment(
        self,
        obstacle: _ObstacleGeometry,
        start: Point,
        end: Point,
    ) -> bool:
        if obstacle.geometry_type == "circle":
            center = obstacle.coordinates[0]
            if obstacle.radius is None:
                raise ValueError("Circle obstacle must provide a radius.")
            return _segment_intersects_circle(start, end, center, obstacle.radius)

        return _segment_intersects_polygon(start, end, obstacle.coordinates)

    def _build_wall_geometries(self) -> List[_ObstacleGeometry]:
        if self.floor_plan is None:
            return []

        walls: List[_ObstacleGeometry] = []
        for element in self.floor_plan.elements:
            if element.element_type != "wall" or element.length is None or element.thickness is None:
                continue

            corners = _rotated_rect_corners(
                anchor_x=element.x,
                anchor_y=element.y,
                length=element.length,
                thickness=element.thickness,
                orientation=element.orientation_angle,
            )
            walls.append(
                _ObstacleGeometry(
                    obstacle_id=element.element_id,
                    obstacle_type="wall",
                    geometry_type="polygon",
                    coordinates=corners,
                    bbox=_bbox_from_points(corners),
                )
            )
        return walls

    def _build_human_geometries(self) -> List[_ObstacleGeometry]:
        geometries: List[_ObstacleGeometry] = []
        for human in self.humans:
            obstacle_id = human.human_label or f"human_{human.human_id}"

            if human.radius is not None:
                center = human.position_X_Y
                radius = human.radius
                geometries.append(
                    _ObstacleGeometry(
                        obstacle_id=obstacle_id,
                        obstacle_type="human",
                        geometry_type="circle",
                        coordinates=(center,),
                        radius=radius,
                        bbox=(
                            center[0] - radius,
                            center[1] - radius,
                            center[0] + radius,
                            center[1] + radius,
                        ),
                    )
                )
                continue

            if human.position_X1_Y1 is None:
                continue

            corners = _human_rect_corners(human)
            geometries.append(
                _ObstacleGeometry(
                    obstacle_id=obstacle_id,
                    obstacle_type="human",
                    geometry_type="polygon",
                    coordinates=corners,
                    bbox=_bbox_from_points(corners),
                )
            )

        return geometries


def build_links(
    antennas: Sequence[Antenna],
    targets: Sequence[Target],
    *,
    floor_plan: Optional[GeneratedFloorPlan] = None,
    humans: Optional[Sequence[Human]] = None,
    scenario_id: Optional[str] = None,
) -> List[Link]:
    return LinkFactory(
        antennas=antennas,
        targets=targets,
        floor_plan=floor_plan,
        humans=humans,
        scenario_id=scenario_id,
    ).build_links()


def build_link_rows(
    antennas: Sequence[Antenna],
    targets: Sequence[Target],
    *,
    floor_plan: Optional[GeneratedFloorPlan] = None,
    humans: Optional[Sequence[Human]] = None,
    scenario_id: Optional[str] = None,
) -> List[Dict[str, object]]:
    return LinkFactory(
        antennas=antennas,
        targets=targets,
        floor_plan=floor_plan,
        humans=humans,
        scenario_id=scenario_id,
    ).build_rows()
