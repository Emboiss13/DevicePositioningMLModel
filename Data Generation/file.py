from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
import math
import random
from typing import Any, Dict, List, Optional, Tuple

# ------------------------------------------------------------------------------
#  TYPES
# ------------------------------------------------------------------------------

# INDOOR_LOS = minimum of 1 completely free channel
# INDOOR_NLOS = not allowed to have any free channels (all must have at least 1 obstacle)
class EnvironmentType(str, Enum):
    INDOOR_LOS = "indoor_LOS"
    INDOOR_NLOS = "indoor_NLOS"
    OUTDOOR = "outdoor"


# We will only create 1 device targer per scenario, the rest will be antennas (can be changed later on)
class DeviceType(str, Enum):
    ANTENNA = "antenna"
    TARGET = "target_endpoint"


class ObstacleType(str, Enum):
    WALL = "wall"                    # max width ~0.3m, length proportional to environment size
    STAIRS = "stairs"                # max width ~3m, max length 3-4m
    HUMAN = "human"                  # radius 0.05-1.5m (standing, sitting, lying down)


# ------------------------------------------------------------------------------
#  GLOBAL HELPERS
# ------------------------------------------------------------------------------

def random_choice(choices: List[Any]) -> Any:
    return random.choice(choices)


def random_point(x_domain: Tuple[float, float], y_range: Tuple[float, float]) -> Tuple[float, float]:
    x0, x1 = x_domain
    y0, y1 = y_range
    return (random.uniform(x0, x1), random.uniform(y0, y1))


def max_amount_given_separation(area_in_m2: float, occupancy_percentage: float, min_separation_in_m: float) -> int:
    area_per_device = math.pi * (min_separation_in_m / 2) ** 2
    usable_area = area_in_m2 * occupancy_percentage
    return max(1, int(usable_area / area_per_device))


def generate_id(counter: int, prefix: str) -> str:
    return f"{prefix}_{counter + 1}"


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# ------------------------------------------------------------------------------
#  ENVIRONMENT
# ------------------------------------------------------------------------------

def random_grid_size(bounds_min: float, bounds_max: float) -> Tuple[float, float]:
    grid_width = random.uniform(bounds_min, bounds_max)
    grid_height = random.uniform(bounds_min, bounds_max)
    return (grid_width, grid_height)


@dataclass
class Environment:
    x_domain: Tuple[float, float]  # meters
    y_range: Tuple[float, float]   # meters
    env_type: EnvironmentType
    grid_area: float               # m^2
    width: float                   # meters
    height: float                  # meters

    def __init__(self) -> None:
        self.env_type = random_choice(list(EnvironmentType))

        bounds_max = 80.0  # 80m x 80m = 6400m^2
        bounds_min = 20.0  # 20m x 20m = 400m^2

        self.width, self.height = random_grid_size(bounds_min, bounds_max)
        self.grid_area = self.width * self.height
        self.x_domain = (-self.width / 2, self.width / 2)
        self.y_range = (-self.height / 2, self.height / 2)


# ------------------------------------------------------------------------------
#  DEVICES
# ------------------------------------------------------------------------------

ENDPOINT_FOOTPRINT_M2 = math.pi * (2.0 / 2.0) ** 2    # radial separation 2m
ANTENNA_FOOTPRINT_M2 = math.pi * (10.0 / 2.0) ** 2    # radial separation 10m


@dataclass
class Device:
    device_id: str
    device_type: DeviceType
    is_target: bool
    position: Tuple[float, float]


class DeviceFactory:
    def __init__(self, env: Environment) -> None:
        self.device_max = round(max_amount_given_separation(env.grid_area, 0.15, 2.0))
        self.antenna_max = round(max_amount_given_separation(env.grid_area, 0.05, 10.0))

        self.antenna_counter: int = 0
        self.target_created: bool = False
        self.x_domain = env.x_domain
        self.y_range = env.y_range

    def create_device(self) -> Device:
        if self.device_counter >= self.device_max:
            raise ValueError(f"Exceeded maximum device count of {self.device_max}")

        # Ensure only one target is created across all devices
        is_target = random_choice([True, False])
        if is_target and self.target_counter < 1:
            self.target_counter += 1
        else:
            is_target = False

        # If we've hit antenna cap, force endpoint
        if self.antenna_counter >= self.antenna_max:
            device_type = DeviceType.ENDPOINT
        else:
            device_type = random_choice(list(DeviceType))

        if device_type == DeviceType.ANTENNA:
            self.antenna_counter += 1

        self.device_counter += 1
        device_id = generate_id(self.device_counter - 1, "device")
        position = random_point(self.x_domain, self.y_range)

        return Device(device_id=device_id, device_type=device_type, is_target=is_target, position=position)



# ------------------------------------------------------------------------------
#  OBSTACLES
# ------------------------------------------------------------------------------

@dataclass
class Obstacle:
    obstacle_id: str
    obstacle_type: ObstacleType
    position_X_Y: Tuple[float, float]
    position_X1_Y1: Optional[Tuple[float, float]]  # rect opposite corner
    radius: Optional[float]                        # humans only
    length: Optional[float]                        # rect only
    width: Optional[float]                         # rect only
    area: float                                   # m^2


class ObstacleFactory:
    def __init__(self, env: Environment) -> None:
        self.obstacle_counter = 0
        self.obstacle_max = round(max_amount_given_separation(env.grid_area, 0.2, 5.0))

        self.x_domain = env.x_domain
        self.y_range = env.y_range
        self.env_width = env.width
        self.env_height = env.height

    def create_obstacle(self) -> Obstacle:
        if self.obstacle_counter >= self.obstacle_max:
            raise ValueError(f"Exceeded maximum obstacle count of {self.obstacle_max}")

        self.obstacle_counter += 1
        obstacle_id = generate_id(self.obstacle_counter - 1, "obstacle")
        obstacle_type = random_choice(list(ObstacleType))

        if obstacle_type == ObstacleType.HUMAN:
            max_radius = 1.5
            min_radius = 0.05
            position = random_point(self.x_domain, self.y_range)
            radius = random.uniform(min_radius, max_radius)
            area = math.pi * radius ** 2
            return Obstacle(
                obstacle_id=obstacle_id,
                obstacle_type=obstacle_type,
                position_X_Y=position,
                position_X1_Y1=None,
                radius=radius,
                length=None,
                width=None,
                area=area,
            )

        if obstacle_type == ObstacleType.STAIRS:
            min_width, max_width = 1.0, 3.0
            min_length, max_length = 2.0, 4.0
            width = random.uniform(min_width, max_width)
            length = random.uniform(min_length, max_length)
            area = width * length
            p0 = random_point(self.x_domain, self.y_range)
            p1 = (p0[0] + width, p0[1] + length)
            return Obstacle(
                obstacle_id=obstacle_id,
                obstacle_type=obstacle_type,
                position_X_Y=p0,
                position_X1_Y1=p1,
                radius=None,
                length=length,
                width=width,
                area=area,
            )

        # WALL
        wall_length_occupancy_percentage = 0.4
        min_width, max_width = 0.25, 0.4
        min_length = 2.0
        max_length = min(self.env_width, self.env_height) * wall_length_occupancy_percentage

        width = random.uniform(min_width, max_width)
        length = random.uniform(min_length, max_length)
        area = width * length
        p0 = random_point(self.x_domain, self.y_range)
        p1 = (p0[0] + width, p0[1] + length)

        return Obstacle(
            obstacle_id=obstacle_id,
            obstacle_type=ObstacleType.WALL,
            position_X_Y=p0,
            position_X1_Y1=p1,
            radius=None,
            length=length,
            width=width,
            area=area,
        )


# ------------------------------------------------------------------------------
#  GEOMETRY + CHANNEL MODEL
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
    minx: float,
    miny: float,
    maxx: float,
    maxy: float,
) -> bool:
    # Liang–Barsky clipping (fast, robust enough here)
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    p = [-dx, dx, -dy, dy]
    q = [a[0] - minx, maxx - a[0], a[1] - miny, maxy - a[1]]
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


def free_space_path_loss_db(distance_m: float, freq_mhz: float) -> float:
    # FSPL(dB) = 32.44 + 20log10(d_km) + 20log10(f_MHz)
    d_km = max(distance_m, 1e-3) / 1000.0
    return 32.44 + 20.0 * math.log10(d_km) + 20.0 * math.log10(freq_mhz)


def env_extra_loss_db(env_type: EnvironmentType, blocking_count: int) -> float:
    # simple knobs; you can tune later
    if env_type == EnvironmentType.OUTDOOR:
        return 2.0 * blocking_count
    if env_type == EnvironmentType.INDOOR_LOS:
        return 4.0 * blocking_count + 8.0
    return 7.0 * blocking_count + 15.0  # INDOOR_NLOS


@dataclass
class Channel:
    device_a_id: str
    device_a_position: Tuple[float, float]
    device_b_id: str
    device_b_position: Tuple[float, float]

    distance_m: float
    freq_mhz: float

    blocking_obstacles: int
    path_loss_db: float
    shadow_fading_db: float
    total_loss_db: float

    # optional "label" for ML use
    link_ok: bool


class ChannelFactory:
    """
    Builds channels for a scenario. By default it connects:
      - every antenna to every endpoint
    Falls back to all-pairs if no antenna exists.
    """
    def __init__(
        self,
        env: Environment,
        obstacles: List[Obstacle],
        *,
        freq_mhz_range: Tuple[float, float] = (2400.0, 5900.0),
        shadow_sigma_db: float = 4.0,
        link_budget_db: float = 120.0,
    ) -> None:
        self.env = env
        self.obstacles = obstacles
        self.freq_mhz_range = freq_mhz_range
        self.shadow_sigma_db = shadow_sigma_db
        self.link_budget_db = link_budget_db

    def _make_channel(self, a: Device, b: Device) -> Channel:
        d = euclidean_distance(a.position, b.position)
        f = random.uniform(*self.freq_mhz_range)
        blocks = obstacles_blocking_count(a.position, b.position, self.obstacles)

        pl = free_space_path_loss_db(d, f)
        extra = env_extra_loss_db(self.env.env_type, blocks)
        shadow = random.gauss(0.0, self.shadow_sigma_db)

        total = pl + extra + shadow
        link_ok = total <= self.link_budget_db

        return Channel(
            device_a_id=a.device_id,
            device_a_position=a.position,
            device_b_id=b.device_id,
            device_b_position=b.position,
            distance_m=d,
            freq_mhz=f,
            blocking_obstacles=blocks,
            path_loss_db=pl,
            shadow_fading_db=shadow,
            total_loss_db=total,
            link_ok=link_ok,
        )

    def build_channels(self, devices: List[Device]) -> List[Channel]:
        antennas = [d for d in devices if d.device_type == DeviceType.ANTENNA]
        endpoints = [d for d in devices if d.device_type == DeviceType.ENDPOINT]

        pairs: List[Tuple[Device, Device]] = []
        if antennas and endpoints:
            for a in antennas:
                for e in endpoints:
                    pairs.append((a, e))
        else:
            # all unique pairs
            for i in range(len(devices)):
                for j in range(i + 1, len(devices)):
                    pairs.append((devices[i], devices[j]))

        return [self._make_channel(a, b) for a, b in pairs]


# ------------------------------------------------------------------------------
#  NETWORK SCENARIO (ORCHESTRATOR)
# ------------------------------------------------------------------------------

def _device_area(d: Device) -> float:
    return ANTENNA_FOOTPRINT_M2 if d.device_type == DeviceType.ANTENNA else ENDPOINT_FOOTPRINT_M2


def _obstacle_area_budget(env: Environment) -> float:
    # maps your comments into actual budgets
    if env.env_type == EnvironmentType.OUTDOOR:
        occ = 0.20
    elif env.env_type == EnvironmentType.INDOOR_LOS:
        occ = 0.35
    else:
        occ = 0.55
    return env.grid_area * occ


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
        min_obstacles: int = 2,
        connect_channels: bool = True,
    ) -> "NetworkScenario":
        """
        Creates a fully randomized scenario:
          1) Environment
          2) Devices (capped by factory max + 15% occupancy intention)
          3) Obstacles (capped by env-type budget + factory max)
          4) Channels (derived from devices + obstacles)
        """
        if seed is not None:
            random.seed(seed)

        env = Environment()

        # ---- Devices
        df = DeviceFactory(env)
        desired_devices = random.randint(min_devices, df.device_max)

        devices: List[Device] = []
        device_area_budget = env.grid_area * 0.15
        device_area_used = 0.0

        while len(devices) < desired_devices:
            d = df.create_device()
            next_area = device_area_used + _device_area(d)
            if next_area > device_area_budget and len(devices) >= min_devices:
                break
            device_area_used = next_area
            devices.append(d)

        # ensure at least one antenna and one endpoint when possible
        if len(devices) >= 2:
            if not any(d.device_type == DeviceType.ANTENNA for d in devices):
                devices[0] = Device(
                    device_id=devices[0].device_id,
                    device_type=DeviceType.ANTENNA,
                    is_target=devices[0].is_target,
                    position=devices[0].position,
                )
            if not any(d.device_type == DeviceType.ENDPOINT for d in devices):
                devices[1] = Device(
                    device_id=devices[1].device_id,
                    device_type=DeviceType.ENDPOINT,
                    is_target=devices[1].is_target,
                    position=devices[1].position,
                )

        # ---- Obstacles
        of = ObstacleFactory(env)
        desired_obstacles = random.randint(min_obstacles, of.obstacle_max)

        obstacles: List[Obstacle] = []
        obstacle_budget = _obstacle_area_budget(env)
        obstacle_area_used = 0.0

        while len(obstacles) < desired_obstacles:
            ob = of.create_obstacle()
            next_area = obstacle_area_used + ob.area
            if next_area > obstacle_budget and len(obstacles) >= min_obstacles:
                break
            obstacle_area_used = next_area
            obstacles.append(ob)

        # ---- Target flag
        target_selected = any(d.is_target for d in devices)

        # ---- Channels
        channels: List[Channel] = []
        if connect_channels and len(devices) >= 2:
            cf = ChannelFactory(env, obstacles)
            channels = cf.build_channels(devices)

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
        JSON-friendly serialization (enums -> values).
        """
        env = asdict(self.environment)
        env["env_type"] = self.environment.env_type.value

        devices = []
        for d in self.devices:
            dd = asdict(d)
            dd["device_type"] = d.device_type.value
            devices.append(dd)

        obstacles = []
        for o in self.obstacles:
            od = asdict(o)
            od["obstacle_type"] = o.obstacle_type.value
            obstacles.append(od)

        channels = [asdict(c) for c in self.channels]

        return {
            "label": self.label,
            "environment": env,
            "target_selected": self.target_selected,
            "devices": devices,
            "obstacles": obstacles,
            "channels": channels,
        }


# ------------------------------------------------------------------------------
#  QUICK MANUAL TEST
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    s = NetworkScenario.generate_random(seed=7)
    d = s.to_dict()
    print("env:", d["environment"]["env_type"], "area:", round(d["environment"]["grid_area"], 2))
    print("devices:", len(d["devices"]), "obstacles:", len(d["obstacles"]), "channels:", len(d["channels"]))
    print("target_selected:", d["target_selected"])
