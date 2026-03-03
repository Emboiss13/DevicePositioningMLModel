from __future__ import annotations
from dataclasses import dataclass
import math
from typing import List, Tuple
from environment_factory import Environment


"""

📡 ANTENNA FACTORY 📡
----------------------

This module includes attributes and methods to generate antennas given environment conditions.

Requirements:
- Antennas must be evenly distributed in the environment.
- Antenna coverage range must cover all areas of the map.
- Antennas must be separated from one another to minimise the amount of antennas needed
  to cover the whole map and maximise the ratio of coverage per antenna.
  
"""



def generate_id(counter: int, prefix: str) -> str:
    return f"{prefix}_{counter}"


@dataclass
class Antenna:
    antenna_id: int
    antenna_label: str
    position: Tuple[float, float]
    coverage_radius: float


class AntennaFactory:
    # Upper/lower practical bounds for a single antenna coverage radius (meters).
    # We need to change this so that we calculate the antenna coverage based on the gain and power
    _MIN_COVERAGE_RADIUS_M = 5.0
    _MAX_COVERAGE_RADIUS_M = 20.0

    def __init__(self, env: Environment) -> None:
        self.env = env
        self.x_domain = env.x_domain
        self.y_range = env.y_range
        self.width = env.width
        self.height = env.height

        self.antenna_counter = 0
        self.antenna_area = 0.2 * 2  # m^2
        self.currently_occupied_antenna_area = 0.0

        rows, cols, radius = self._coverage_grid_plan()
        self.coverage_radius = radius
        self._planned_positions = self._grid_positions(rows, cols)
        self._planned_count = len(self._planned_positions)

        # Cap exactly at the planned coverage-complete deployment.
        self.allowed_antenna_area = self._planned_count * self.antenna_area

    def _required_radius_for_grid(self, rows: int, cols: int) -> float:
        cell_w = self.width / cols
        cell_h = self.height / rows
        # Radius needed so each cell corner is still covered by its cell-center antenna.
        return 0.5 * math.sqrt(cell_w * cell_w + cell_h * cell_h)

    def _coverage_grid_plan(self) -> Tuple[int, int, float]:
        # Find minimum antennas (rows*cols) that can still satisfy max coverage radius.
        # Ties are broken by choosing bigger required radius (less overlap, better ratio/antenna).
        best_rows = 0
        best_cols = 0
        best_required_radius = 0.0
        best_count: int | None = None

        max_dim = max(2, int(math.ceil(max(self.width, self.height))))
        for rows in range(1, max_dim + 1):
            for cols in range(1, max_dim + 1):
                required_radius = self._required_radius_for_grid(rows, cols)
                if required_radius > self._MAX_COVERAGE_RADIUS_M:
                    continue

                count = rows * cols
                is_better_count = best_count is None or count < best_count
                is_better_overlap = best_count is not None and count == best_count and required_radius > best_required_radius
                if is_better_count or is_better_overlap:
                    best_rows = rows
                    best_cols = cols
                    best_required_radius = required_radius
                    best_count = count

        if best_count is None:
            # Fallback: derive a guaranteed-feasible grid directly from max radius.
            max_cell_side = self._MAX_COVERAGE_RADIUS_M * math.sqrt(2.0)
            best_cols = max(1, math.ceil(self.width / max_cell_side))
            best_rows = max(1, math.ceil(self.height / max_cell_side))
            best_required_radius = self._required_radius_for_grid(best_rows, best_cols)

        radius = max(self._MIN_COVERAGE_RADIUS_M, best_required_radius)
        return best_rows, best_cols, radius

    def _grid_positions(self, rows: int, cols: int) -> List[Tuple[float, float]]:
        x0, x1 = self.x_domain
        y0, y1 = self.y_range
        cell_w = (x1 - x0) / cols
        cell_h = (y1 - y0) / rows

        positions: List[Tuple[float, float]] = []
        for r in range(rows):
            for c in range(cols):
                x = x0 + (c + 0.5) * cell_w
                y = y0 + (r + 0.5) * cell_h
                positions.append((x, y))
        return positions

    def has_capacity(self) -> bool:
        return self.antenna_counter < self._planned_count

    def create_antenna(self) -> Antenna:
        if not self.has_capacity():
            raise RuntimeError("Full-map coverage plan already reached.")

        projected_area = self.currently_occupied_antenna_area + self.antenna_area
        if projected_area > self.allowed_antenna_area + 1e-9:
            raise ValueError("Exceeded maximum allowed area for antennas")

        antenna_id = self.antenna_counter
        antenna_label = generate_id(antenna_id, "antenna")
        position = self._planned_positions[antenna_id]

        self.antenna_counter += 1
        self.currently_occupied_antenna_area = min(projected_area, self.allowed_antenna_area)

        return Antenna(
            antenna_id=antenna_id,
            antenna_label=antenna_label,
            position=position,
            coverage_radius=self.coverage_radius,
        )
