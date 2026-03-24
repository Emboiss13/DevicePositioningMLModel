"""
TARGET FACTORY
--------------

This module takes in an environment, a grid for said env and places a target in every cell of the grid. 

The output of this module should be a dict with the key being the grid_id and the value the coordinate of the target in that cell. 

@author: Giuliana Emberson
@date: 7th of May 2026

"""

from dataclasses import dataclass
from typing import Tuple

@dataclass
class Target:
    target_id: int
    target_label: str
    position: Tuple[float, float]