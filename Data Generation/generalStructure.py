from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import math
from typing import List, Tuple, Optional, TypeVar, Type
import random

# ------------------------------------------------------------------------------
#  TYPES
# ------------------------------------------------------------------------------

E = TypeVar("E", bound=Enum)

class EnvironmentType(str, Enum):
    INDOOR = "indoor"
    OUTDOOR = "outdoor"


class DeviceType(str, Enum):
    ANTENNA = "antenna"
    ENDPOINT = "endpoint"  # mobile/handset/etc.


class ObstacleType(str, Enum):
    WALL = "wall"
    STAIRS = "stairs"
    FURNITURE = "furniture"
    HUMAN = "human"

# ------------------------------------------------------------------------------
# 🌍 GLOBAL HELPER FUNCTIONS 🌍
# ------------------------------------------------------------------------------

def random_enum(enum_cls: Type[E]) -> E:
    return random.choice(list(enum_cls))

# Generate a random point within given bounds
def random_point(x_bounds: Tuple[float, float], y_bounds: Tuple[float, float]) -> Tuple[float, float]:
    x0, x1 = x_bounds
    y0, y1 = y_bounds
    return (random.uniform(x0, x1), random.uniform(y0, y1))

def max_amount_given_separation(area_in_m2: float, occupancy_percentage: float, min_separation_in_m: float) -> int:
    # Approximate each device as a disk with radius = min_separation_in_m/2
    area_per_device = math.pi * (min_separation_in_m / 2) ** 2
    usable_area = area_in_m2 * occupancy_percentage
    return max(1, int(usable_area / area_per_device))

# ------------------------------------------------------------------------------
# 🌳 ENVIRONMENT HELPER FUNCTIONS 🌳
# ------------------------------------------------------------------------------

# Grid size bounds (meters)
def random_grid_size(bounds_min: float, bounds_max: float) -> Tuple[float, float]:
    grid_width = random.uniform(bounds_min, bounds_max)
    grid_height = random.uniform(bounds_min, bounds_max)
    return (grid_width, grid_height)

@dataclass
class Environment:
    x_bounds: Tuple[float, float] # in m
    y_bounds: Tuple[float, float] # in m
    env_type: EnvironmentType
    grid_area: float # in m^2

    def __init__(self):
        self.env_type = random_enum(EnvironmentType)

        
        bounds_max = 80.0 # 80m x 80m = 6400 m^2
        bounds_min = 20.0 # 20m x 20m = 400 m^2
        
        width, height = random_grid_size(bounds_min, bounds_max)
        self.grid_area = width * height
        self.x_bounds = (-width / 2, width / 2)
        self.y_bounds = (-height / 2, height / 2)

# ------------------------------------------------------------------------------
# 📡 DEVICES HELPER FUNCTIONS 🖥️
# ------------------------------------------------------------------------------

def generate_device_id(device_counter: int, device_max: int) -> str:
    if device_counter >= device_max:
        raise ValueError(f"Exceeded maximum device count of {device_max}")
    return f"device_{device_counter + 1}"

def random_device_type() -> DeviceType:
    return random.choice(list(DeviceType))


@dataclass
class Device:
    device_id: str
    device_type: DeviceType
    position: Tuple[float, float]

class DeviceFactory:
    def __init__(self, env: Environment):

        # 10% of the area can be occupied by devices / antennas must be at least 2m apart
        self.device_max = round(max_amount_given_separation(env.grid_area, 0.1, 2.0))
        
        # 5% of the area can be occupied by antennas / antennas must be at least 10m apart
        self.antenna_max = round(max_amount_given_separation(env.grid_area, 0.05, 10.0))

        self.device_counter = 0
        self.antenna_counter = 0
        self.x_bounds = env.x_bounds
        self.y_bounds = env.y_bounds

    def create_device(self) -> Device:

        # Only create a new device if the max hasn't been reached yet
        if self.device_counter >= self.device_max:
            raise ValueError(f"Exceeded maximum device count of {self.device_max}")

        device_type = random_enum(DeviceType)
        if device_type == DeviceType.ANTENNA:
            if self.antenna_counter >= self.antenna_max:
                raise ValueError(f"Exceeded maximum antenna count of {self.antenna_max}")
            self.antenna_counter += 1

        self.device_counter += 1

        device_id = generate_device_id(self.device_counter - 1, self.device_max)

        position = random_point(self.x_bounds, self.y_bounds)

        return Device(device_id=device_id, device_type=device_type, position=position)

# ------------------------------------------------------------------------------
# 🧱 OBSTACLES HELPER FUNCTIONS 🪴
# ------------------------------------------------------------------------------

@dataclass
class Obstacle:
    obstacle_id: str
    obstacle_type: ObstacleType
    position: Tuple[float, float]
    size_max: Tuple[float, float] # for rectangles: (max_w, max_h), for circles: (max_diameter, 0)
    size: Tuple[float, float]  # rectangle w,h OR (diameter, 0) for circles


class ObstacleFactory:
    def __init__(self, env: Environment) -> Obstacle:

        # 20% of the area can be occupied by obstacles
        self.obstacle_max = round(max_amount_given_separation(env.grid_area, 0.2, 5.0))

        self.obstacle_counter = 0
        self.x_bounds = env.x_bounds
        self.y_bounds = env.y_bounds

    def create_obstacle(self) -> Obstacle:

        # Only create a new obstacle if the max hasn't been reached yet
        if self.obstacle_counter >= self.obstacle_max:
            raise ValueError(f"Exceeded maximum obstacle count of {self.obstacle_max}")

        self.obstacle_counter += 1

        obstacle_id = f"obstacle_{self.obstacle_counter}"
        obstacle_type = random_enum(ObstacleType)
        position = random_point(self.x_bounds, self.y_bounds)

        if obstacle_type == ObstacleType.HUMAN:
            # Humans will be represented as circles
            size_max = (1.0, 1.0) # max width and height of human obstacles; this allows for a variety of sizes up to that point
            size = (random.uniform(0.5, size_max[0]), random.uniform(0.5, size_max[1]))
        
        elif obstacle_type == ObstacleType.STAIRS:
            # Stairs will be represented as wide rectangles
            size_max = (5.0, 5.0) # max width and height of stairs obstacles; this allows for a variety of sizes up to that point
            size = (random.uniform(2.0, size_max[0]), random.uniform(2.0, size_max[1]))
        
        elif obstacle_type == ObstacleType.WALL:
            # Walls can be long and thin rectangles
            size_max = (20.0, 1.0) # max width and height of wall obstacles; this allows for a variety of sizes up to that point
            size = (random.uniform(5.0, size_max[0]), random.uniform(0.5, size_max[1]))
        
        else:
            # Furniture can be represented as rectangles of various sizes
            size_max = (10.0, 10.0) # max width and height of obstacles; this allows for a variety of sizes up to that point
            size = (random.uniform(1.0, size_max[0]), random.uniform(1.0, size_max[1]))

        return Obstacle(obstacle_id=obstacle_id, obstacle_type=obstacle_type, position=position, size_max=size_max, size=size)


    

    
    
    

@dataclass
class Channel:
    device_a_id: str
    device_b_id: str
    distance: float

@dataclass
class NetworkScenario:
    label: str
    environment: Environment
    target_selected: bool
    devices: List[Device] = field(default_factory=list)
    obstacles: List[Obstacle] = field(default_factory=list)
    channels: List[Channel] = field(default_factory=list)





# Using Euclidean distance for simplicity; can be extended to more complex models
def euclidean_distance(point_1: Tuple[float, float], point_2: Tuple[float, float]) -> float:
    dx = point_1[0] - point_2[0]
    dy = point_1[1] - point_2[1]
    return (dx * dx + dy * dy) ** 0.5

