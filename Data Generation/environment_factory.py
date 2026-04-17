"""
ENVIRONMENR FACTORY
-------------------

This module includes attributes and methods to randomly generate network environments.

Requirements: 
- The grid cannot be larger than 6400 m^2
- The grid cannot be smaller than 400 m^2

@author: Giuliana Emberson
@date: 7th of May 2026

"""

import random
from typing import Optional, Tuple
from dataclasses import dataclass


# Grid size bounds (meters)
def random_grid_size(bounds_min: float, bounds_max: float) -> Tuple[float, float]:
    grid_width = random.uniform(bounds_min, bounds_max)
    grid_height = random.uniform(bounds_min, bounds_max)
    return (grid_width*2, grid_height*2)

@dataclass
class Environment:
    x_domain: Tuple[float, float]  # meters
    y_range: Tuple[float, float]   # meters
    width: float                   # meters
    height: float                  # meters
    area: float                    # m^2
    env_type: str                  # "indoor" or "outdoor"
    
    def __init__(self, *, env_type: Optional[str] = None):
        # Use larger environment bounds so that realistic 5 GHz coverage radius
        # (80–230 m) require multiple antennas to cover the full map.
        #
        # Recommended bounds (meters):
        #   - Min: 100 (200×200 = 40,000 m^2)
        #   - Max: 400 (800×800 = 640,000 m^2)
        #
        # This keeps the environments large enough to need more than one
        # antenna while keeping the scenario computationally manageable.
        bounds_min = 100
        bounds_max = 400

        self.width, self.height = random_grid_size(bounds_min, bounds_max)

        self.x_domain = (-self.width / 2, self.width / 2)
        self.y_range = (-self.height / 2, self.height / 2)
        
        self.area = self.width * self.height
        self.env_type = env_type or random.choice(["indoor", "outdoor"])
