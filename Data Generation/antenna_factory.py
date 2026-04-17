"""
ANTENNA FACTORY
---------------

This module defines the attributes and helper methods used to generate Clear-Com FreeSpeak Edge transceiver antennas for synthetic network scenarios.

Generation assumptions:
- Antennas should be distributed across the environment to provide broad coverage.
- Antenna placement should account for realistic 5 GHz propagation behaviour.
- Antennas should be sufficiently separated to reduce excessive overlap while still covering the intended area.
- Coverage radius may be sampled within realistic ranges instead of being derived from first principles.

Hardware constraints (based on Clear-Com FreeSpeak Edge FSE-TCVR-50-IP):
- Antenna / transceiver type: External omnidirectional 5 GHz transceiver antenna 
- Frequency spectrum: 5170-5875 MHz 
- Channel width: 20 MHz
- Modulation: OFDM
- Antenna gain: 3 dB omni
- Output power: Adjustable from 1 to 24 dBm
- Dimensions without bracket (W x H x D): 193.04 x 209.6 x 85.6 mm
- Dimensions with bracket (W x H x D): 193.04 x 209.6 x 104.65 mm

Practical coverage guidance from datasheet:
- Outdoor LOS coverage: 160-230 m
- Indoor coverage: 80-107 m
- Reflective surfaces increase coverage distance

NOTE:
- FreeSpeak Edge operates in the 5 GHz spectrum but is not standard Wi-Fi.
- Depending on region, the 5 GHz band provides 25+ non-overlapping channels.
- For simulation, antenna radius can be randomly selected within realistic deployment bounds consistent with the datasheet and scenario type.


Antenna generation pseudocode: 
1) Randomly chose if we are creating a Outdoor scenario or Indoor one. 
2) Based on environment type set range. 
3) Randomly select a fixed antenna coverage given the environment type coverage range. 
4) Generate antennas by evenly spacing them out in the area based on their coverage range, so no part of the map is left uncovered. 


@author: Giuliana Emberson
@date: 7th of May 2026

"""

from __future__ import annotations
from dataclasses import dataclass
import math
import random
from typing import List, Tuple
from environment_factory import Environment


def generate_id(counter: int, prefix: str) -> str:
    return f"{prefix}_{counter}"


@dataclass
class Antenna:
    antenna_id: int                          # unique numeric ID
    antenna_label: str                       # antenna label (for plotting/visualisation)
    position: Tuple[float, float]            # (x,y) coordinates
    coverage_radius: float                   # same for all antennas in that scenario


class AntennaFactory:
    # Coverage ranges based on typical 5Ghz FSE-TCVR-50-IP FreeSpeak Edge coverage documentation.
    _OUTDOOR_COVERAGE_RANGE_M = (160.0, 230.0)
    _INDOOR_COVERAGE_RANGE_M = (80.0, 107.0)

    def __init__(self, env: Environment) -> None:
        self.env = env
        self.env_x_domain = env.x_domain
        self.env_y_range = env.y_range
        self.env_width = env.width
        self.env_height = env.height

        self.antenna_counter = 0

        rows, cols, radius = self._coverage_grid_plan()
        self.coverage_radius = radius
        self._planned_positions = self._grid_positions(rows, cols)
        self._planned_count = len(self._planned_positions)

    def _coverage_grid_plan(self) -> Tuple[int, int, float]:
        """Plan antenna placement so that the full environment is covered.

        1) Use the scenario environment type to determine whether coverage is indoor or outdoor.
        2) Select a fixed antenna coverage radius within the corresponding range.
        3) Compute a grid size that ensures no part of the map is left uncovered.
        """
        env_type = getattr(self.env, "env_type", None)
        if env_type not in {"indoor", "outdoor"}:
            env_type = random.choice(["indoor", "outdoor"])

        self.environment_type = env_type
        self.is_outdoor = env_type == "outdoor"

        min_radius, max_radius = (
            self._OUTDOOR_COVERAGE_RANGE_M
            if self.is_outdoor
            else self._INDOOR_COVERAGE_RANGE_M
        )
        radius = random.uniform(min_radius, max_radius)

        # For a given antenna radius, the maximum square cell side that can be fully
        # covered from its center is radius * sqrt(2). Choosing this cell size ensures
        # coverage even when the grid is non-square.
        max_cell_side = radius * math.sqrt(2.0)

        cols = max(1, math.ceil(self.env_width / max_cell_side))
        rows = max(1, math.ceil(self.env_height / max_cell_side))

        return rows, cols, radius

    def _grid_positions(self, rows: int, cols: int) -> List[Tuple[float, float]]:
        x0, x1 = self.env_x_domain
        y0, y1 = self.env_y_range
        cell_w = (x1 - x0) / cols
        cell_h = (y1 - y0) / rows

        positions: List[Tuple[float, float]] = []
        for r in range(rows):
            for c in range(cols):
                x = x0 + (c + 0.5) * cell_w
                y = y0 + (r + 0.5) * cell_h
                positions.append((x, y))
        return positions

    def get_antenna_positions(self) -> List[Tuple[float, float]]:
        """Return a stable list of antenna coordinates for this scenario.

        This allows downstream processing (e.g., RSSI/TDOA/AOA calculations)
        to use the exact same antenna layout without recomputing the grid.
        """
        return list(self._planned_positions)

    def has_capacity(self) -> bool:
        return self.antenna_counter < self._planned_count

    def create_antenna(self) -> Antenna:
        if not self.has_capacity():
            raise RuntimeError("Full-map coverage plan already reached.")

        antenna_id = self.antenna_counter
        antenna_label = generate_id(antenna_id, "antenna")
        position = self._planned_positions[antenna_id]

        self.antenna_counter += 1

        return Antenna(
            antenna_id=antenna_id,
            antenna_label=antenna_label,
            position=position,
            coverage_radius=self.coverage_radius,
        )
