from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import math
import random
from typing import List, Optional, Tuple
from environment_factory import Environment


"""

🧱 OBSTACLE FACTORY 👩🏻‍💻
----------------------

This module includes attributes and methods to generate obstacles given environment conditions.

Requirements: 
- Human-human and structural-structural spacing must be at least 2m.
- There is a max occupied area for obstacles vs free space in the environment. 

"""



class ObstacleType(str, Enum):
    WALL = "wall"                    # max width ~0.3m, max length proportional to environment size and occupied area
    STAIRS = "stairs"                # fixed width of 3m and length of 4m
    HUMAN = "human"                  # radius 0.5 m (standing, sitting, lying down)

    
# Check max area hasn't been reached
def currently_occupied_obstacle_area_is_valid(currently_occupied_obstacle_area: float, area: float, allowed_obstacle_area: float) -> bool:

    if currently_occupied_obstacle_area + area >= allowed_obstacle_area:
        return False
    else:
        return True


# Min occupied area = 10% 
# Max occupied area = 70%
def allowed_obstacle_area() -> float:
    return random.uniform(0.1, 0.7)


# Generate a random point ensuring the shape of size (dx, dy) fully fits the bounds
def random_point_for_rect(x_domain: Tuple[float, float], y_range: Tuple[float, float], dx: float, dy: float) -> Tuple[float, float]:
    x0, x1 = x_domain
    y0, y1 = y_range
    if dx > (x1 - x0) or dy > (y1 - y0):
        raise ValueError("Rectangle does not fit within the provided domain")
    return (random.uniform(x0, x1 - dx), random.uniform(y0, y1 - dy))


def random_point_for_circle(x_domain: Tuple[float, float], y_range: Tuple[float, float], radius: float) -> Tuple[float, float]:
    x0, x1 = x_domain
    y0, y1 = y_range
    if (radius * 2) > (x1 - x0) or (radius * 2) > (y1 - y0):
        raise ValueError("Circle does not fit within the provided domain")
    return (random.uniform(x0 + radius, x1 - radius), random.uniform(y0 + radius, y1 - radius))


@dataclass
class Obstacle:
    obstacle_id: int
    obstacle_label: str
    obstacle_type: ObstacleType
    position_X_Y: Tuple[float, float]
    position_X1_Y1: Optional[Tuple[float, float]]     # rect opposite corner
    radius: Optional[float]                           # humans only
    length: Optional[float]                           # rect only
    width: Optional[float]                            # rect only
    area: float                                       # m^2


class ObstacleFactory:
    _MIN_OBSTACLE_SEPARATION_M = 2.0
    _TYPE_SELECTION_WEIGHT = {
        ObstacleType.HUMAN: 0.70,
        ObstacleType.WALL: 0.20,
        ObstacleType.STAIRS: 0.10,
    }

    def __init__(self, env: Environment) -> None:
        
        self.env = env
        self.obstacle_counter = 0
        self.x_domain = env.x_domain
        self.y_range = env.y_range

        self.allowed_obstacle_area = env.grid_area * allowed_obstacle_area()
        self.currently_occupied_obstacle_area = 0.0
        self._count_by_type = {t: 0 for t in ObstacleType}
        base_span = min(self.env.width, self.env.height)
        self._max_count_by_type = {
            ObstacleType.HUMAN: 10**9,
            ObstacleType.WALL: max(2, int(base_span / 12.0)),
            ObstacleType.STAIRS: max(1, int(base_span / 25.0)),
        }

        self._placed: List[Obstacle] = []

    def _remaining_area(self) -> float:
        return self.allowed_obstacle_area - self.currently_occupied_obstacle_area

    def _can_place(self, area: float) -> bool:
        return currently_occupied_obstacle_area_is_valid(
            self.currently_occupied_obstacle_area, area, self.allowed_obstacle_area
        )

    def _next_obstacle_id(self) -> int:
        self.obstacle_counter += 1
        return self.obstacle_counter - 1

    def _min_area_for_type(self, t: ObstacleType) -> float:
        if t == ObstacleType.HUMAN:
            #50 cm = 0.5 m
            return math.pi * (0.5 ** 2)

        if t == ObstacleType.STAIRS:
            min_width, min_length = 1.0, 2.0
            return min_width * min_length

        if t == ObstacleType.WALL:
            min_width = 0.25
            base_span = min(self.env.width, self.env.height)   # meters
            min_length = 0.10 * base_span                      # 10% of smaller side
            return min_width * min_length

        raise ValueError(f"Unsupported obstacle type: {t}")

    def _global_min_possible_area(self) -> float:
        return min(self._min_area_for_type(t) for t in ObstacleType)

    def can_fit_any_obstacle(self) -> bool:
        return self._remaining_area() >= self._global_min_possible_area()

    # --- Overlap helper functions ---
    def _rect_bounds_from_points(self, p0: Tuple[float, float], p1: Tuple[float, float]) -> Tuple[float, float, float, float]:
        x0, y0 = p0
        x1, y1 = p1
        minx, maxx = (x0, x1) if x0 <= x1 else (x1, x0)
        miny, maxy = (y0, y1) if y0 <= y1 else (y1, y0)
        return minx, maxx, miny, maxy

    def _rect_edge_distance(
        self,
        r1: Tuple[float, float, float, float],
        r2: Tuple[float, float, float, float],
    ) -> float:
        minx1, maxx1, miny1, maxy1 = r1
        minx2, maxx2, miny2, maxy2 = r2
        dx = max(minx2 - maxx1, minx1 - maxx2, 0.0)
        dy = max(miny2 - maxy1, miny1 - maxy2, 0.0)
        return math.sqrt(dx * dx + dy * dy)

    def _circle_edge_distance(
        self,
        c1: Tuple[float, float],
        r1: float,
        c2: Tuple[float, float],
        r2: float,
    ) -> float:
        dx = c1[0] - c2[0]
        dy = c1[1] - c2[1]
        center_distance = math.sqrt(dx * dx + dy * dy)
        return center_distance - (r1 + r2)

    def _circle_rect_edge_distance(
        self,
        center: Tuple[float, float],
        radius: float,
        rect: Tuple[float, float, float, float],
    ) -> float:
        minx, maxx, miny, maxy = rect
        cx, cy = center
        dx = max(minx - cx, 0.0, cx - maxx)
        dy = max(miny - cy, 0.0, cy - maxy)
        return math.sqrt(dx * dx + dy * dy) - radius

    def _edge_distance_between(self, candidate: Obstacle, existing: Obstacle) -> float:
        if candidate.radius is not None and existing.radius is not None:
            return self._circle_edge_distance(
                candidate.position_X_Y,
                candidate.radius,
                existing.position_X_Y,
                existing.radius,
            )

        if candidate.position_X1_Y1 and existing.position_X1_Y1:
            candidate_rect = self._rect_bounds_from_points(candidate.position_X_Y, candidate.position_X1_Y1)
            existing_rect = self._rect_bounds_from_points(existing.position_X_Y, existing.position_X1_Y1)
            return self._rect_edge_distance(candidate_rect, existing_rect)

        circ = candidate if candidate.radius is not None else existing
        rect_ob = existing if candidate.radius is not None else candidate
        rect_bounds = self._rect_bounds_from_points(rect_ob.position_X_Y, rect_ob.position_X1_Y1)  # type: ignore[arg-type]
        return self._circle_rect_edge_distance(circ.position_X_Y, circ.radius, rect_bounds)  # type: ignore[arg-type]

    def _is_structural(self, obstacle: Obstacle) -> bool:
        return obstacle.obstacle_type in {ObstacleType.WALL, ObstacleType.STAIRS}

    def _required_spacing(self, candidate: Obstacle, existing: Obstacle) -> float:
        if candidate.obstacle_type == ObstacleType.HUMAN and existing.obstacle_type == ObstacleType.HUMAN:
            return self._MIN_OBSTACLE_SEPARATION_M
        if self._is_structural(candidate) and self._is_structural(existing):
            return self._MIN_OBSTACLE_SEPARATION_M
        return 0.0

    def _overlaps_existing(self, candidate: Obstacle) -> bool:
        for ob in self._placed:
            if self._edge_distance_between(candidate, ob) < self._required_spacing(candidate, ob):
                return True
        return False

    def _record(self, obstacle: Obstacle) -> Obstacle:
        self._placed.append(obstacle)
        self.currently_occupied_obstacle_area += obstacle.area
        self._count_by_type[obstacle.obstacle_type] += 1
        return obstacle

    def _candidate_types(self) -> List[ObstacleType]:
        remaining_area = self._remaining_area()
        fit_types = [
            t
            for t in ObstacleType
            if remaining_area >= self._min_area_for_type(t)
            and self._count_by_type[t] < self._max_count_by_type[t]
        ]
        if not fit_types:
            return []

        ordered_types: List[ObstacleType] = []
        pool = fit_types.copy()
        while pool:
            weights = [self._TYPE_SELECTION_WEIGHT[t] for t in pool]
            selected = random.choices(pool, weights=weights, k=1)[0]
            ordered_types.append(selected)
            pool.remove(selected)
        return ordered_types

    def _build_obstacle_label(self, obstacle_type: ObstacleType, obstacle_id: int) -> str:
        return f"{obstacle_type.value}_{obstacle_id}"

    def create_obstacle(self) -> Obstacle:
        if not self.can_fit_any_obstacle():
            raise RuntimeError("No obstacle can fit in the remaining allowed area.")

        obstacle_types = self._candidate_types()
        if not obstacle_types:
            raise RuntimeError("No obstacle type can fit in remaining area.")

        for obstacle_type in obstacle_types:
            # Position retry loop to avoid overlaps without an infinite loop
            for _ in range(30):
                if obstacle_type == ObstacleType.HUMAN:
                    # min_radius, max_radius = 0.05, 1.5
                    # radius = random.uniform(min_radius, max_radius)
                    radius = 0.5
                    human_area = math.pi * radius**2

                    if not self._can_place(human_area):
                        continue

                    position = random_point_for_circle(self.x_domain, self.y_range, radius)
                    obstacle_id = self._next_obstacle_id()
                    candidate = Obstacle(
                        obstacle_label=self._build_obstacle_label(obstacle_type, obstacle_id),
                        obstacle_id=obstacle_id,
                        obstacle_type=obstacle_type,
                        position_X_Y=position,
                        position_X1_Y1=None,
                        radius=radius,
                        length=None,
                        width=None,
                        area=human_area,
                    )
                    if self._overlaps_existing(candidate):
                        continue
                    return self._record(candidate)

                if obstacle_type == ObstacleType.STAIRS:
                    min_width, max_width = 1.0, 3.0
                    min_length, max_length = 2.0, 4.0
                    width = random.uniform(min_width, max_width)
                    length = random.uniform(min_length, max_length)
                    stairs_area = width * length

                    if not self._can_place(stairs_area):
                        continue

                    p0 = random_point_for_rect(self.x_domain, self.y_range, width, length)
                    p1 = (p0[0] + width, p0[1] + length)
                    obstacle_id = self._next_obstacle_id()
                    candidate = Obstacle(
                        obstacle_label=self._build_obstacle_label(obstacle_type, obstacle_id),
                        obstacle_id=obstacle_id,
                        obstacle_type=obstacle_type,
                        position_X_Y=p0,
                        position_X1_Y1=p1,
                        radius=None,
                        length=length,
                        width=width,
                        area=stairs_area,
                    )
                    if self._overlaps_existing(candidate):
                        continue
                    return self._record(candidate)

                if obstacle_type == ObstacleType.WALL:
                    base_span = min(self.env.width, self.env.height)
                    min_width, max_width = 0.25, 0.4
                    min_length, max_length = 0.10 * base_span, 0.50 * base_span
                    width = random.uniform(min_width, max_width)
                    length = random.uniform(min_length, max_length)
                    wall_area = width * length


                    if not self._can_place(wall_area):
                        continue

                    p0 = random_point_for_rect(self.x_domain, self.y_range, width, length)
                    p1 = (p0[0] + width, p0[1] + length)
                    obstacle_id = self._next_obstacle_id()
                    candidate = Obstacle(
                        obstacle_label=self._build_obstacle_label(obstacle_type, obstacle_id),
                        obstacle_id=obstacle_id,
                        obstacle_type=obstacle_type,
                        position_X_Y=p0,
                        position_X1_Y1=p1,
                        radius=None,
                        length=length,
                        width=width,
                        area=wall_area,
                    )
                    if self._overlaps_existing(candidate):
                        continue
                    return self._record(candidate)
            # Exhausted attempts for this type; move on to next type

        raise RuntimeError(
            "At least one type should fit by minimum-area check, but random draw did not fit this call. Try again."
        )
