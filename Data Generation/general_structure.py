from __future__ import annotations
from dataclasses import asdict, dataclass, field
from enum import Enum
import math
from typing import Any, Dict, List, Tuple, Optional, TypeVar, Type
import random

"""
------------------------------------------------------------------------------
TOTAL OCCUPIED AREA CONSTRAINTS
------------------------------------------------------------------------------

DEVICES
 - Max 15% of the area can be occupied by devices
 - We need the antennas to fill the space to ensure connectivity, but they also need to be spaced out to create a realistic scenario

OBSTACLES
 - OUTDOOR: Max 20% (free-space = 80% - 15% devices = 65%)
 - INDOOR LOS: Max 30-40% (free-space = 70-60% - 15% devices = 55-45%)
 - INDOOR NLOS: Max 50-60% (free-space = 50-40% - 15% devices = 35-25%)

"""

# ------------------------------------------------------------------------------
#  TYPES
# ------------------------------------------------------------------------------

class EnvironmentType(str, Enum):
    INDOOR_LOS = "indoor_LOS"
    INDOOR_NLOS = "indoor_NLOS"
    OUTDOOR = "outdoor"


# We will only create 1 device targer per scenario, the rest will be antennas (can be changed later on)
class DeviceType(str, Enum):
    ANTENNA = "antenna"
    TARGET = "target_endpoint"


class ObstacleType(str, Enum):
    WALL = "wall"                    # max width ~0.3m, max length proportional to environment size
    STAIRS = "stairs"                # max width ~3m, max length 3-4m
    HUMAN = "human"                  # radius 0.05-1.5m (standing, sitting, lying down)

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

def check_available_space(available_space: float, new_obstacle_area: float) -> bool:
    return new_obstacle_area <= available_space

def generate_id(counter: int, prefix: str) -> str:
    return f"{prefix}_{counter + 1}"

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

# ------------------------------------------------------------------------------
# 🌳 ENVIRONMENT 🌳
# ------------------------------------------------------------------------------

# Grid size bounds (meters)
def random_grid_size(bounds_min: float, bounds_max: float) -> Tuple[float, float]:
    grid_width = random.uniform(bounds_min, bounds_max)
    grid_height = random.uniform(bounds_min, bounds_max)
    return (grid_width*2, grid_height*2)

def bounds_given_type(env_type: EnvironmentType) -> Tuple[float, float]:
    if env_type == EnvironmentType.OUTDOOR:
        return (60.0, 80.0)
    elif env_type == EnvironmentType.INDOOR_LOS:
        return (20.0, 70.0)
    elif env_type == EnvironmentType.INDOOR_NLOS:
        return (10.0, 50.0)

@dataclass
class Environment:
    x_domain: Tuple[float, float]  # meters
    y_range: Tuple[float, float]   # meters
    env_type: EnvironmentType
    grid_area: float               # m^2
    width: float                   # meters
    height: float                  # meters

    def __init__(self):
        self.env_type = random_choice(list(EnvironmentType))

        random_bounds = bounds_given_type(self.env_type)
        bounds_max = random_bounds[1]
        bounds_min = random_bounds[0]
        
        self.width, self.height = random_grid_size(bounds_min, bounds_max)
        self.grid_area = self.width * self.height
        self.x_domain = (-self.width / 2, self.width / 2)
        self.y_range = (-self.height / 2, self.height / 2)

# ------------------------------------------------------------------------------
# 📡 DEVICES 🖥️
# ------------------------------------------------------------------------------

@dataclass
class Device:
    device_id: str
    device_type: DeviceType
    is_target: bool
    position: Tuple[float, float]

class DeviceFactory:
    """
    Builds Obstacles for a scenario considering the following range and domain constraints:

    - Outdoor = 60–80 m          --> 80m x 80m = 6400 m^2
    - Indoor LOS: 20–70 m        --> 20m x 20m = 400 m^2, 70m x 70m = 4900 m^2
    - Indoor NLOS: 10–50 m       --> 10m x 10m = 100 m^2, 50m x 50m = 2500 m^2
    """
    def __init__(self, env: Environment) -> None:

        self.allowed_device_area = env.grid_area * 0.15 # 15% of the area can be occupied by
        self.currently_occupied_device_area: float = 0
        self.x_domain = env.x_domain
        self.y_range = env.y_range
        self.target_created: bool = False
        self.device_counter: int = 0
        self.device_area: float = 0.2*2  # m²

    def create_device(self) -> Device:
        if self.currently_occupied_device_area + self.device_area > self.allowed_device_area:
            raise ValueError("Exceeded maximum allowed area for devices")

        self.currently_occupied_device_area += self.device_area
        self.device_counter += 1

        if not self.target_created:
            is_target = True
            device_type = DeviceType.TARGET
            self.target_created = True
        else:
            is_target = False
            device_type = DeviceType.ANTENNA

        device_id = generate_id(self.device_counter - 1, "device")
        position = random_point(self.x_domain, self.y_range)
        return Device(device_id=device_id, device_type=device_type, is_target=is_target, position=position)



# ------------------------------------------------------------------------------
# 🧱 OBSTACLES 🪴
# ------------------------------------------------------------------------------

def allowed_obstacle_area(env_type: EnvironmentType) -> float:
    # return FRACTION of grid area
    if env_type == EnvironmentType.OUTDOOR:
        return 0.20
    if env_type == EnvironmentType.INDOOR_LOS:
        return random.uniform(0.30, 0.40)
    if env_type == EnvironmentType.INDOOR_NLOS:
        return random.uniform(0.50, 0.60)
    raise ValueError(f"Unknown environment type: {env_type}")
    
# Check max area hasn't been reached
def currently_occupied_obstacle_area_is_valid(currently_occupied_obstacle_area: float, area: float, allowed_obstacle_area: float) -> bool:

    if currently_occupied_obstacle_area + area >= allowed_obstacle_area:
        return False
    else:
        return True


@dataclass
class Obstacle:
    obstacle_id: str
    obstacle_type: ObstacleType
    position_X_Y: Tuple[float, float]
    position_X1_Y1: Optional[Tuple[float, float]]    # rect opposite corner
    radius: Optional[float]                           # humans only
    length: Optional[float]                           # rect only
    width: Optional[float]                            # rect only
    area: float                                       # m^2


class ObstacleFactory:
    """
    Builds Obstacles for a scenario considering the following constraints: 

    - OUTDOOR: Max space occupied by obstacles 20% (free-space = 80% - 15% devices = 65%)
    - INDOOR_LOS: Max space occupied by obstacles 30-40% (free-space = 70-60% - 15% devices = 55-45%)
    - INDOOR_NLOS: Max space occupied by obstacles 50-60% (free-space = 50-40% - 15% devices = 35-25%)
    """
    def __init__(self, env: Environment) -> None:
        self.env = env
        self.obstacle_counter = 0
        # absolute m^2 budget = fraction * grid_area
        self.allowed_obstacle_area = env.grid_area * allowed_obstacle_area(env.env_type)
        self.currently_occupied_obstacle_area = 0.0
        self.x_domain = env.x_domain
        self.y_range = env.y_range

    def _remaining_area(self) -> float:
        return self.allowed_obstacle_area - self.currently_occupied_obstacle_area

    def _can_place(self, area: float) -> bool:
        return currently_occupied_obstacle_area_is_valid(
            self.currently_occupied_obstacle_area, area, self.allowed_obstacle_area
        )

    def _next_obstacle_id(self) -> str:
        self.obstacle_counter += 1
        return generate_id(self.obstacle_counter - 1, "obstacle")

    def _min_area_for_type(self, t: ObstacleType) -> float:
        if t == ObstacleType.HUMAN:
            min_radius = 0.05
            return math.pi * (min_radius ** 2)

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

    def create_obstacle(self) -> Obstacle:
        if not self.can_fit_any_obstacle():
            raise RuntimeError("No obstacle can fit in the remaining allowed area.")

        # Try each type once in random order; avoids invalid 'continue' outside loop.
        obstacle_types = list(ObstacleType)
        random.shuffle(obstacle_types)

        for obstacle_type in obstacle_types:
            if obstacle_type == ObstacleType.HUMAN:
                min_radius, max_radius = 0.05, 1.5
                radius = random.uniform(min_radius, max_radius)
                human_area = math.pi * radius**2

                if not self._can_place(human_area):
                    continue

                position = random_point(self.x_domain, self.y_range)
                obstacle_id = self._next_obstacle_id()
                self.currently_occupied_obstacle_area += human_area

                return Obstacle(
                    obstacle_id=obstacle_id,
                    obstacle_type=obstacle_type,
                    position_X_Y=position,
                    position_X1_Y1=None,
                    radius=radius,
                    length=None,
                    width=None,
                    area=human_area,
                )

            if obstacle_type == ObstacleType.STAIRS:
                min_width, max_width = 1.0, 3.0
                min_length, max_length = 2.0, 4.0
                width = random.uniform(min_width, max_width)
                length = random.uniform(min_length, max_length)
                stairs_area = width * length

                if not self._can_place(stairs_area):
                    continue

                p0 = random_point(self.x_domain, self.y_range)
                p1 = (p0[0] + width, p0[1] + length)
                obstacle_id = self._next_obstacle_id()
                self.currently_occupied_obstacle_area += stairs_area

                return Obstacle(
                    obstacle_id=obstacle_id,
                    obstacle_type=obstacle_type,
                    position_X_Y=p0,
                    position_X1_Y1=p1,
                    radius=None,
                    length=length,
                    width=width,
                    area=stairs_area,
                )

            if obstacle_type == ObstacleType.WALL:
                base_span = min(self.env.width, self.env.height)
                min_width, max_width = 0.25, 0.4
                min_length, max_length = 0.10 * base_span, 0.50 * base_span
                width = random.uniform(min_width, max_width)
                length = random.uniform(min_length, max_length)
                wall_area = width * length


                if not self._can_place(wall_area):
                    continue

                p0 = random_point(self.x_domain, self.y_range)
                p1 = (p0[0] + width, p0[1] + length)
                obstacle_id = self._next_obstacle_id()
                self.currently_occupied_obstacle_area += wall_area

                return Obstacle(
                    obstacle_id=obstacle_id,
                    obstacle_type=obstacle_type,
                    position_X_Y=p0,
                    position_X1_Y1=p1,
                    radius=None,
                    length=length,
                    width=width,
                    area=wall_area,
                )

        raise RuntimeError(
            "At least one type should fit by minimum-area check, but random draw did not fit this call. Try again."
        )



# ------------------------------------------------------------------------------
# 🎛️ CHANNELS 📶
# ------------------------------------------------------------------------------

def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return (dx * dx + dy * dy) ** 0.5


def _dot(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _sub(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    return (a[0] - b[0], a[1] - b[1])


def _segment_intersects_aabb(
    a: Tuple[float, float],
    b: Tuple[float, float],
    min_x: float,
    min_y: float,
    max_x: float,
    max_y: float,
) -> bool:
    # Liang–Barsky clipping algorithm for line-AABB intersection
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    p = [-dx, dx, -dy, dy]
    q = [a[0] - min_x, max_x - a[0], a[1] - min_y, max_y - a[1]]
    u1, u2 = 0.0, 1.0

    for pi, qi in zip(p, q):
        if pi == 0:
            if qi < 0:
                return False
        else:
            t = qi / pi
            if pi < 0:
                u1 = max(u1, t)
            else:
                u2 = min(u2, t)
            if u1 > u2:
                return False
    return True


def _segment_intersects_circle(
    a: Tuple[float, float],
    b: Tuple[float, float],
    center: Tuple[float, float],
    radius: float,
) -> bool:
    # distance from center to segment <= radius
    ab = _sub(b, a)
    ac = _sub(center, a)
    ab_len2 = _dot(ab, ab)
    if ab_len2 == 0:
        return euclidean_distance(a, center) <= radius

    t = _dot(ac, ab) / ab_len2
    t = clamp(t, 0.0, 1.0)
    closest = (a[0] + ab[0] * t, a[1] + ab[1] * t)
    return euclidean_distance(closest, center) <= radius


def obstacles_blocking_count(
    a: Tuple[float, float],
    b: Tuple[float, float],
    obstacles: List[Obstacle],
) -> int:
    count = 0
    for ob in obstacles:
        if ob.obstacle_type == ObstacleType.HUMAN and ob.radius is not None:
            if _segment_intersects_circle(a, b, ob.position_X_Y, ob.radius):
                count += 1
        else:
            if ob.position_X1_Y1 is None:
                continue
            x0, y0 = ob.position_X_Y
            x1, y1 = ob.position_X1_Y1
            minx, maxx = (x0, x1) if x0 <= x1 else (x1, x0)
            miny, maxy = (y0, y1) if y0 <= y1 else (y1, y0)
            if _segment_intersects_aabb(a, b, minx, miny, maxx, maxy):
                count += 1
    return count


@dataclass
class Channel:
    device_a_id: str
    device_a_position: Tuple[float, float]
    device_b_id: str
    device_b_position: Tuple[float, float]
    distance_m: float
    freq_mhz: float
    blocking_obstacles: int


class ChannelFactory:
    """
    Builds channels for a scenario.

    NOTE:
    - INDOOR_LOS = minimum of 1 completely free channel
    - INDOOR_NLOS = not allowed to have any free channels (all must have at least 1 obstacle)
    """
    def __init__(
        self,
        env: Environment,
        obstacles: List[Obstacle],
        *,
        freq_ghz: float = 5.0,
    ) -> None:
        self.env = env
        self.obstacles = obstacles
        self.freq_ghz = freq_ghz

    def _make_channel(self, a: Device, b: Device) -> Channel:
        d = euclidean_distance(a.position, b.position)
        f = self.freq_ghz * 1000.0  # convert to MHz
        blocks = obstacles_blocking_count(a.position, b.position, self.obstacles)

        return Channel(
            device_a_id=a.device_id,
            device_a_position=a.position,
            device_b_id=b.device_id,
            device_b_position=b.position,
            distance_m=d,
            freq_mhz=f,
            blocking_obstacles=blocks,
        )

    def build_channels(self, devices: List[Device]) -> List[Channel]:
        antennas = [d for d in devices if d.device_type == DeviceType.ANTENNA]
        targets = [d for d in devices if d.device_type == DeviceType.TARGET]


        pairs: List[Tuple[Device, Device]] = []
        if antennas and targets:
            for a in antennas:
                for t in targets:
                    pairs.append((a, t))
        else:
            # all unique pairs
            for i in range(len(devices)):
                for j in range(i + 1, len(devices)):
                    pairs.append((devices[i], devices[j]))

        return [self._make_channel(a, b) for a, b in pairs]
    

def channels_meet_env_constraints(env_type: EnvironmentType, channels: List[Channel]) -> bool:
    """
    Enforce per-environment LOS/NLOS rules:
    - INDOOR_LOS: at least one channel must have 0 blocking obstacles (completely free).
    - INDOOR_NLOS: no channel may be completely free (each must have >=1 blocker).
    - OUTDOOR: no constraint.
    """
    if not channels:
        return True  # nothing to validate

    if env_type == EnvironmentType.INDOOR_LOS:
        return any(ch.blocking_obstacles == 0 for ch in channels)

    if env_type == EnvironmentType.INDOOR_NLOS:
        return all(ch.blocking_obstacles >= 1 for ch in channels)

    return True  # OUTDOOR or future envs


"""
A seed was included for the cases where I want to corroborate that the calculations are """

@dataclass
class NetworkScenario:
    label: str
    environment: Environment
    target_selected: bool
    devices: List[Device] = field(default_factory=list)
    obstacles: List[Obstacle] = field(default_factory=list)
    channels: List[Channel] = field(default_factory=list)

    @classmethod
    def generate_random(
        cls,
        *,
        label: str = "random_scenario",
        seed: Optional[int] = None,
        min_devices: int = 4,
        connect_channels: bool = True,
        max_channel_regen_attempts: int = 10,
    ) -> "NetworkScenario":
        """
        Creates a randomized scenario:
        1) Environment
        2) Devices (capped by factory max + 15% occupancy intention)
        3) Obstacles (until area can no longer fit or attempt cap reached)
        4) Channels (derived from devices + obstacles)
        """
        if seed is not None:
            random.seed(seed)

        env = Environment()

        # ---- Devices
        device_factory = DeviceFactory(env)
        devices: List[Device] = []

        # Random target count without relying on df.device_max
        desired_devices = min_devices + random.randint(0, 20)

        while len(devices) < desired_devices:
            try:
                d = device_factory.create_device()
            except Exception:
                break  # factory can't place more devices

            devices.append(d)

        # ---- Obstacles
        obstacle_factory = ObstacleFactory(env)
        obstacles: List[Obstacle] = []
        failed_draws = 0
        max_failed_draws = 50  # safety only

        while obstacle_factory.can_fit_any_obstacle():
            try:
                obstacle = obstacle_factory.create_obstacle()
                obstacles.append(obstacle)
                failed_draws = 0  # reset after success
            except RuntimeError:
                failed_draws += 1
                if failed_draws >= max_failed_draws:
                    break


        # ---- Target flag
        target_selected = any(d.is_target for d in devices)

        # ---- Channels
        channels: List[Channel] = []
        if connect_channels and len(devices) >= 2:
            channel_factory = ChannelFactory(env, obstacles)
            channels = channel_factory.build_channels(devices)

            # Regenerate obstacles (and thus channels) if LOS/NLOS constraints are not met
            attempts = 0
            while not channels_meet_env_constraints(env.env_type, channels):
                attempts += 1
                if attempts >= max_channel_regen_attempts:
                    break

                # recreate obstacles and channels
                obstacle_factory = ObstacleFactory(env)
                obstacles = []
                while obstacle_factory.can_fit_any_obstacle():
                    try:
                        obstacle = obstacle_factory.create_obstacle()
                        obstacles.append(obstacle)
                    except RuntimeError:
                        break

                channel_factory = ChannelFactory(env, obstacles)
                channels = channel_factory.build_channels(devices)


        return cls(
            label=label,
            environment=env,
            target_selected=target_selected,
            devices=devices,
            obstacles=obstacles,
            channels=channels,
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        JSON-style formatting for generated data.
        """
        env = asdict(self.environment)
        env["env_type"] = self.environment.env_type.value

        devices = []
        for device in self.devices:
            device_dict = asdict(device)
            device_dict["device_type"] = device.device_type.value
            devices.append(device_dict)

        obstacles = []
        for obstacle in self.obstacles:
            obstacle_dict = asdict(obstacle)
            obstacle_dict["obstacle_type"] = obstacle.obstacle_type.value
            obstacles.append(obstacle_dict)

        channels = [asdict(channel) for channel in self.channels]

        return {
            "label": self.label,
            "environment": env,
            "target_selected": self.target_selected,
            "devices": devices,
            "obstacles": obstacles,
            "channels": channels,
        }