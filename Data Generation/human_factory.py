"""
HUMAN FACTORY
-------------

This module includes attributes and methods to generate humans given environment conditions.

Requirements:
- Minimum separation of 2 meters between humans to avoid excessive clustering.
- Humans should be randomly distributed across the environment while respecting the separation constraint.
- Humans should be placed within corridors and rooms, meaning not overlaping the floorplan.

How does this link to the other modules in the env generation?:
1) Generate floor plan -> Walls, doors, windows, stairs.
2) Generate humans -> Inside these spaces

NOTE: When deciding on the separation a new human must have from the next, instead of having a fixed value, should we randomly chose a distance?
        - Can the distance be calculated relative to the total available free area?
        - How would that even be justifiable?
        - What other ways do we have of doing this?

@author: Giuliana Emberson
@date: 7th of May 2026
"""

from __future__ import annotations
from dataclasses import dataclass
import math
import random
from enum import Enum
from typing import Optional, Tuple
from environment_factory import Environment



def random_point_for_circle(
    x_domain: Tuple[float, float],
    y_range: Tuple[float, float],
    radius: float,
) -> Tuple[float, float]:
    x0, x1 = x_domain
    y0, y1 = y_range

    if (radius * 2) > (x1 - x0) or (radius * 2) > (y1 - y0):
        raise ValueError("Circle does not fit within the provided domain")

    return (
        random.uniform(x0 + radius, x1 - radius),
        random.uniform(y0 + radius, y1 - radius),
    )


"""
# Make sure to implement next
class HumanType(str, Enum):
    SITTING = "sitting"
    LAYING_DOWN = "layinig_down"
    STANDING = "standing"
"""


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


class HumanFactory:
    _HUMAN_RADIUS_M = 5.0

    def __init__(self, env: Environment) -> None:
        self.env = env
        self.human_counter = 0
        self._human_created = False

    def can_fit_any_human(self) -> bool:
        return not self._human_created

    def _next_human_id(self) -> int:
        human_id = self.human_counter
        self.human_counter += 1
        return human_id

    def _build_human_label(self, human_id: int) -> str:
        return f"human_{human_id}"

    def create_human(self) -> Human:
        if not self.can_fit_any_human():
            raise RuntimeError("Test human has already been created for this environment.")

        radius = self._HUMAN_RADIUS_M
        position = random_point_for_circle(self.env.x_domain, self.env.y_range, radius)
        human_id = self._next_human_id()

        human = Human(
            human_id=human_id,
            human_label=self._build_human_label(human_id),
            position_X_Y=position,
            position_X1_Y1=None,
            radius=radius,
            length=None,
            width=None,
            area=math.pi * (radius ** 2),
        )

        self._human_created = True
        return human
