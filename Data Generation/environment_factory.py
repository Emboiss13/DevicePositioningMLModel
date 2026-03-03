import random
from typing import Tuple
from dataclasses import dataclass


"""

🌳 ENVIRONMENR FACTORY 🏠
-------------------------

This module includes attributes and methods to randomly generate network environments.

Requirements: 
- The grid cannot be larger than 6400 m^2
- The grid cannot be smaller than 400 m^2

"""



# Grid size bounds (meters)
def random_grid_size(bounds_min: float, bounds_max: float) -> Tuple[float, float]:
    grid_width = random.uniform(bounds_min, bounds_max)
    grid_height = random.uniform(bounds_min, bounds_max)
    return (grid_width*2, grid_height*2)

@dataclass
class Environment:
    grid_x_domain: Tuple[float, float]  # meters
    grid_y_range: Tuple[float, float]   # meters
    grid_spaces: int
    
    env_id: int                        # E.g. 1
    env_label: str                     # E.g. "Environment_1"
    env_area: float                    # m^2
    env_width: float                   # meters
    env_height: float                  # meters

    def __init__(self):
        bounds_max = 80                # 80m x 80m = 6400 m^2
        bounds_min = 20                # 20m x 20m = 400 m^2
        self.width, self.height = random_grid_size(bounds_min, bounds_max)
        self.grid_area = self.width * self.height
        self.x_domain = (-self.width / 2, self.width / 2)
        self.y_range = (-self.height / 2, self.height / 2)