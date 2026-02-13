from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, List, Tuple, Optional, TypeVar, Type
import random

# ------------------------------------------------------------------------------
#  TOTAL OCCUPIED AREA CONSTRAINTS
# ------------------------------------------------------------------------------

# DEVICES = 15%
# - Max 10% of the area can be occupied by endpoints
# - Max 5% of the area can be occupied by antennas

# OBSTACLES
# - OUTDOOR: Max 20% (free-space = 80% - 15% devices = 65%)
# - INDOOR LOS: Max 30-40% (free-space = 70-60% - 15% devices = 55-45%)
# - INDOOR NLOS: Max 50-60% (free-space = 50-40% - 15% devices = 35-25%)


# ------------------------------------------------------------------------------
#  TYPES
# ------------------------------------------------------------------------------

class EnvironmentType(str, Enum):
    INDOOR_LOS = "indoor_LOS"
    INDOOR_NLOS = "indoor_NLOS"
    OUTDOOR = "outdoor"


class DeviceType(str, Enum):
    ANTENNA = "antenna"
    ENDPOINT = "endpoint"  # mobile/handset/etc.


class ObstacleType(str, Enum):
    WALL = "wall"
    STAIRS = "stairs"
    HUMAN = "human"

# ------------------------------------------------------------------------------
# 🌍 GLOBAL HELPER FUNCTIONS 🌍
# ------------------------------------------------------------------------------

def random_choice(choices: List[Any]) -> Any:
    return random.choice(choices)

# Generate a random point within given bounds
def random_point(x_domain: Tuple[float, float], y_range: Tuple[float, float]) -> Tuple[float, float]:
    x0, x1 = x_domain
    y0, y1 = y_range
    return (random.uniform(x0, x1), random.uniform(y0, y1))

def max_amount_given_separation(area_in_m2: float, occupancy_percentage: float, min_separation_in_m: float) -> int:
    # Approximate each device as a disk with radius = min_separation_in_m/2
    area_per_device = math.pi * (min_separation_in_m / 2) ** 2
    usable_area = area_in_m2 * occupancy_percentage
    return max(1, int(usable_area / area_per_device))

def generate_id(counter: int, prefix: str) -> str:
    return f"{prefix}_{counter + 1}"

# ------------------------------------------------------------------------------
# 🌳 ENVIRONMENT 🌳
# ------------------------------------------------------------------------------

# Grid size bounds (meters)
def random_grid_size(bounds_min: float, bounds_max: float) -> Tuple[float, float]:
    grid_width = random.uniform(bounds_min, bounds_max)
    grid_height = random.uniform(bounds_min, bounds_max)
    return (grid_width, grid_height)

@dataclass
class Environment:
    x_domain: Tuple[float, float] # in m
    y_range: Tuple[float, float] # in m
    env_type: EnvironmentType
    grid_area: float # in m^2

    def __init__(self):
        self.env_type = random_choice(list(EnvironmentType))

        
        bounds_max = 80.0 # 80m x 80m = 6400 m^2
        bounds_min = 20.0 # 20m x 20m = 400 m^2
        
        width, height = random_grid_size(bounds_min, bounds_max)
        self.grid_area = width * height
        self.x_domain = (-width / 2, width / 2)
        self.y_range = (-height / 2, height / 2)

# ------------------------------------------------------------------------------
# 📡 DEVICES 🖥️
# ------------------------------------------------------------------------------

ENDPOINT_FOOTPRINT_M2 = math.pi * (2.0 / 2.0) ** 2   # min separation 2m
ANTENNA_FOOTPRINT_M2 = math.pi * (10.0 / 2.0) ** 2   # min separation 10m

@dataclass
class Device:
    device_id: str
    device_type: DeviceType
    is_target:bool = False
    position: Tuple[float, float]

class DeviceFactory:
    def __init__(self, env: Environment):

        # 15% of the area can be occupied by devices / antennas must be at least 2m apart
        self.device_max = round(max_amount_given_separation(env.grid_area, 0.15, 2.0))
        
        # 5% of the area can be occupied by antennas / antennas must be at least 10m apart
        self.antenna_max = round(max_amount_given_separation(env.grid_area, 0.05, 10.0))

        self.device_counter: int = 0
        self.antenna_counter: int = 0
        self.target_counter: int = 0
        self.x_domain: Tuple[float, float] = env.x_domain
        self.y_range: Tuple[float, float] = env.y_range

    def create_device(self) -> Device:

        # Only create a new device if the max hasn't been reached yet
        if self.device_counter >= self.device_max:
            raise ValueError(f"Exceeded maximum device count of {self.device_max}")
        
        # Ensure only one target is created across all devices
        is_target = random_choice([True, False])
        if is_target and self.target_counter < 1:
            self.target_counter += 1
        else:
            is_target = False

        # Randomly select device type, but ensure we don't exceed the max antenna count
        device_type = random_choice(list(DeviceType))
        if device_type == DeviceType.ANTENNA:
            if self.antenna_counter >= self.antenna_max:
                raise ValueError(f"Exceeded maximum antenna count of {self.antenna_max}")
            self.antenna_counter += 1

        self.device_counter += 1

        device_id = generate_id(self.device_counter - 1, "device")

        position = random_point(self.x_domain, self.y_range)

        return Device(device_id=device_id, device_type=device_type, is_target=is_target, position=position)

# ------------------------------------------------------------------------------
# 🧱 OBSTACLES 🪴
# ------------------------------------------------------------------------------

@dataclass
class Obstacle:
    obstacle_id: str
    obstacle_type: ObstacleType
    position_X_Y: Tuple[float, float]
    position_X1_Y1: Optional[Tuple[float, float]] # for walls & stairs only; represents the opposite corner of the rectangle
    radius: Optional[float] # for humans only
    length: Optional[float] # for walls & stairs only
    width: Optional[float] # for walls & stairs only
    area: float # in m^2


class ObstacleFactory:
    def __init__(self, env: Environment) -> Obstacle:

        self.obstacle_counter = 0

        # 20% of the area can be occupied by obstacles
        self.obstacle_max = round(max_amount_given_separation(env.grid_area, 0.2, 5.0))
        
        self.x_domain = env.x_domain
        self.y_range = env.y_range

    def create_obstacle(self) -> Obstacle:

        # Only create a new obstacle if the max hasn't been reached yet
        if self.obstacle_counter >= self.obstacle_max:
            raise ValueError(f"Exceeded maximum obstacle count of {self.obstacle_max}")

        self.obstacle_counter += 1

        obstacle_id = generate_id(self.obstacle_counter - 1, "obstacle")
        obstacle_type = random_choice(list(ObstacleType))

        # Humans will be represented as circles
        if obstacle_type == ObstacleType.HUMAN:

            max_radius = 1.5 # meters
            min_radius = 0.05 # meters

            position_X_Y = random_point(self.x_domain, self.y_range)
            radius = random.uniform(min_radius, max_radius)
            area = math.pi * radius ** 2

            return Obstacle(obstacle_id=obstacle_id, obstacle_type=obstacle_type, position_X_Y=position_X_Y, position_X1_Y1=None, radius=radius, length=None, width=None, area=area)
        
        # Stairs will be represented as wide rectangles
        elif obstacle_type == ObstacleType.STAIRS:
            
            # Constraints for dimenssions
            min_width = 1.0 # meters
            max_width = 3.0 # meters
            min_length = 2.0 # meters
            max_length = 4.0 # meters

            # Randomly generate position and size
            width = random.uniform(min_width, max_width)
            length = random.uniform(min_length, max_length)
            area = width * length
            position_X_Y = random_point(self.x_domain, self.y_range)
            position_X1_Y1 = (position_X_Y[0] + width, position_X_Y[1] + length)
            
            return Obstacle(obstacle_id=obstacle_id, obstacle_type=obstacle_type, position_X_Y=position_X_Y, position_X1_Y1=position_X1_Y1, radius=None, length=length, width=width, area=area)
        
        # Walls will be long and thin rectangles
        elif obstacle_type == ObstacleType.WALL:
            # Constraints for dimenssions
            wall_length_occupancy_percentage = 0.4 # walls can occupy up to 80% of the obstacle space
            min_width = 0.25 # meters
            max_width = 0.4 # meters
            min_length = 2.0 # meters
            max_length = min(self.x_domain, self.y_range)* wall_length_occupancy_percentage # meters

            # Randomly generate position and size
            width = random.uniform(min_width, max_width)
            length = random.uniform(min_length, max_length)
            area = width * length
            position_X_Y = random_point(self.x_domain, self.y_range)
            position_X1_Y1 = (position_X_Y[0] + width, position_X_Y[1] + length)
            
            return Obstacle(obstacle_id=obstacle_id, obstacle_type=obstacle_type, position_X_Y=position_X_Y, position_X1_Y1=position_X1_Y1, radius=None, length=length, width=width, area=area)


# ------------------------------------------------------------------------------
# 🎛️ CHANNELS 📶
# ------------------------------------------------------------------------------

# Using Euclidean distance for simplicity; can be extended to more complex models
def euclidean_distance(point_1: Tuple[float, float], point_2: Tuple[float, float]) -> float:
    dx = point_1[0] - point_2[0]
    dy = point_1[1] - point_2[1]
    return (dx * dx + dy * dy) ** 0.5

@dataclass
class Channel:
    device_a_id: str
    device_a_position: Tuple[float, float]
    device_b_id: str
    device_b_position: Tuple[float, float]
    distance: float

@dataclass
class NetworkScenario:
    label: str
    environment: Environment
    target_selected: bool
    devices: List[Device] = field(default_factory=list)
    obstacles: List[Obstacle] = field(default_factory=list)
    channels: List[Channel] = field(default_factory=list)


