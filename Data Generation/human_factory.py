"""
HUMAN FACTORY
-------------

This module includes attributes and methods to generate humans given environment
conditions and an already-generated floor plan.

Requirements:
- Minimum separation of 2 meters between humans to avoid excessive clustering.
- Humans should be randomly distributed while respecting the separation
  constraint.
- Humans should be placed inside rooms and corridors, not on walls or patios.

How this links to the other modules:
1) Generate floor plan -> Walls, doors, windows, patios.
2) Generate humans -> Inside the generated rooms/corridors.

@author: Giuliana Emberson
@date: 7th of May 2026
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Dict, List, Optional, Tuple

from environment_factory import Environment
from floor_plan_factory import GeneratedFloorPlan, Rect, Room


def random_point_for_circle(
    x_domain: Tuple[float, float],
    y_range: Tuple[float, float],
    radius: float,
    *,
    rng: Optional[random.Random] = None,
) -> Tuple[float, float]:
    local_rng = rng or random
    x0, x1 = x_domain
    y0, y1 = y_range

    if (radius * 2) > (x1 - x0) or (radius * 2) > (y1 - y0):
        raise ValueError("Circle does not fit within the provided domain")

    return (
        local_rng.uniform(x0 + radius, x1 - radius),
        local_rng.uniform(y0 + radius, y1 - radius),
    )


@dataclass
class Human:
    human_id: int
    human_label: str
    position_X_Y: Tuple[float, float]
    position_X1_Y1: Optional[Tuple[float, float]]
    radius: Optional[float]
    length: Optional[float]
    width: Optional[float]
    area: float
    room_id: Optional[str] = None
    room_type: Optional[str] = None


@dataclass(frozen=True)
class RoomPlacementPlan:
    room_id: str
    room_type: str
    usable_rect: Rect
    max_capacity: int


class HumanFactory:
    _HUMAN_RADIUS_M = 0.35
    _WALL_CLEARANCE_M = 0.4
    _MIN_HUMAN_SEPARATION_M = 2.0
    _MAX_ATTEMPTS_PER_ROOM = 120
    _ROOM_TYPE_CAPACITY_LIMITS: Dict[str, int] = {
        "small": 1,
        "medium": 2,
        "large": 4,
        "corridor": 2,
    }
    _ROOM_OCCUPANCY_RANGES: Dict[str, Dict[str, Tuple[float, float]]] = {
        "indoor": {
            "small": (0.05, 0.85),
            "medium": (0.10, 0.90),
            "large": (0.15, 0.95),
            "corridor": (0.00, 0.45),
        },
        "outdoor": {
            "small": (0.00, 0.50),
            "medium": (0.05, 0.65),
            "large": (0.10, 0.75),
            "corridor": (0.00, 0.25),
        },
    }
    _ROOM_PRESENCE_PROBABILITY: Dict[str, Dict[str, float]] = {
        "indoor": {
            "small": 0.30,
            "medium": 0.45,
            "large": 0.60,
            "corridor": 0.20,
        },
        "outdoor": {
            "small": 0.15,
            "medium": 0.30,
            "large": 0.40,
            "corridor": 0.08,
        },
    }
    _SCENARIO_ROOM_OCCUPANCY_RATIO: Dict[str, Tuple[float, float]] = {
        "indoor": (0.08, 0.18),
        "outdoor": (0.03, 0.10),
    }

    def __init__(
        self,
        env: Environment,
        floor_plan: Optional[GeneratedFloorPlan] = None,
        *,
        random_seed: Optional[int] = None,
    ) -> None:
        self.env = env
        self.floor_plan = floor_plan
        self.human_counter = 0
        self._human_created = False
        self.rng = random.Random(random_seed)

    def can_fit_any_human(self) -> bool:
        return not self._human_created

    def _next_human_id(self) -> int:
        human_id = self.human_counter
        self.human_counter += 1
        return human_id

    def _build_human_label(self, human_id: int) -> str:
        return f"human_{human_id}"

    def _build_circular_human(
        self,
        position: Tuple[float, float],
        *,
        room_id: Optional[str] = None,
        room_type: Optional[str] = None,
    ) -> Human:
        human_id = self._next_human_id()
        radius = self._HUMAN_RADIUS_M
        return Human(
            human_id=human_id,
            human_label=self._build_human_label(human_id),
            position_X_Y=position,
            position_X1_Y1=None,
            radius=radius,
            length=None,
            width=None,
            area=math.pi * (radius ** 2),
            room_id=room_id,
            room_type=room_type,
        )

    def create_human(self) -> Human:
        if not self.can_fit_any_human():
            raise RuntimeError("Test human has already been created for this environment.")

        position = random_point_for_circle(
            self.env.x_domain,
            self.env.y_range,
            self._HUMAN_RADIUS_M,
            rng=self.rng,
        )
        human = self._build_circular_human(position)
        self._human_created = True
        return human

    def generate_humans(self) -> List[Human]:
        if self.floor_plan is None:
            return [self.create_human()]

        placement_rooms = self._build_room_placement_plans()
        if not placement_rooms:
            return []

        scenario_cap = self._sample_scenario_human_cap(len(placement_rooms))
        if scenario_cap <= 0:
            return []

        humans: List[Human] = []
        ordered_rooms = sorted(
            placement_rooms,
            key=lambda room: room.usable_rect.area * self.rng.uniform(0.9, 1.1),
            reverse=True,
        )

        for room in ordered_rooms:
            remaining = scenario_cap - len(humans)
            if remaining <= 0:
                break

            room_target = min(self._sample_room_target(room), remaining)
            for position in self._sample_positions_for_room(room, room_target):
                humans.append(
                    self._build_circular_human(
                        position,
                        room_id=room.room_id,
                        room_type=room.room_type,
                    )
                )

        return humans

    def _sample_scenario_human_cap(self, room_count: int) -> int:
        ratio_min, ratio_max = self._SCENARIO_ROOM_OCCUPANCY_RATIO[self.env.env_type]
        return int(round(room_count * self.rng.uniform(ratio_min, ratio_max)))

    def _build_room_placement_plans(self) -> List[RoomPlacementPlan]:
        if self.floor_plan is None:
            return []

        placement_plans: List[RoomPlacementPlan] = []
        for room in self.floor_plan.rooms:
            usable_rect = self._usable_room_rect(room)
            if usable_rect is None:
                continue

            max_capacity = self._room_capacity(room, usable_rect)
            if max_capacity <= 0:
                continue

            placement_plans.append(
                RoomPlacementPlan(
                    room_id=room.room_id,
                    room_type=room.room_type,
                    usable_rect=usable_rect,
                    max_capacity=max_capacity,
                )
            )
        return placement_plans

    def _usable_room_rect(self, room: Room) -> Optional[Rect]:
        margin = self._HUMAN_RADIUS_M + self._WALL_CLEARANCE_M
        usable_width = room.rect.width - (2 * margin)
        usable_height = room.rect.height - (2 * margin)
        if usable_width <= 0 or usable_height <= 0:
            return None

        return Rect(
            x_min=room.rect.x_min + margin,
            y_min=room.rect.y_min + margin,
            x_max=room.rect.x_max - margin,
            y_max=room.rect.y_max - margin,
        )

    def _room_capacity(self, room: Room, usable_rect: Rect) -> int:
        center_spacing = (2 * self._HUMAN_RADIUS_M) + self._MIN_HUMAN_SEPARATION_M
        cols = max(1, int(usable_rect.width // center_spacing) + 1)
        rows = max(1, int(usable_rect.height // center_spacing) + 1)
        capacity = cols * rows
        return min(capacity, self._ROOM_TYPE_CAPACITY_LIMITS.get(room.room_type, 2))

    def _sample_room_target(self, room: RoomPlacementPlan) -> int:
        if room.max_capacity <= 0:
            return 0

        room_presence = self._ROOM_PRESENCE_PROBABILITY[self.env.env_type].get(room.room_type, 0.25)
        if self.rng.random() > room_presence:
            return 0

        range_min, range_max = self._ROOM_OCCUPANCY_RANGES[self.env.env_type].get(room.room_type, (0.0, 0.5))
        raw_target = room.max_capacity * self.rng.uniform(range_min, range_max)
        target = int(raw_target)
        if self.rng.random() < (raw_target - target):
            target += 1
        if target == 0:
            target = 1
        return min(target, room.max_capacity)

    def _sample_positions_for_room(
        self,
        room: RoomPlacementPlan,
        target: int,
    ) -> List[Tuple[float, float]]:
        if target <= 0:
            return []

        positions: List[Tuple[float, float]] = []
        attempts = 0
        max_attempts = max(self._MAX_ATTEMPTS_PER_ROOM, target * 60)

        while len(positions) < target and attempts < max_attempts:
            candidate = (
                self.rng.uniform(room.usable_rect.x_min, room.usable_rect.x_max),
                self.rng.uniform(room.usable_rect.y_min, room.usable_rect.y_max),
            )
            if self._position_is_valid(candidate, positions):
                positions.append(candidate)
            attempts += 1

        return positions

    def _position_is_valid(
        self,
        candidate: Tuple[float, float],
        existing_positions: List[Tuple[float, float]],
    ) -> bool:
        min_center_distance = (2 * self._HUMAN_RADIUS_M) + self._MIN_HUMAN_SEPARATION_M
        for existing in existing_positions:
            if math.dist(candidate, existing) < min_center_distance:
                return False
        return True
