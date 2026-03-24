from typing import Tuple, List, Optional
from human_factory import Human
from matplotlib.pylab import Enum

"""

RSSI LINK FACTORY
----------------

This module includes attributes and methods to generate RSSI links given environment conditions, positions of antennas in the map and the current position of the target in the environment.

"""

class RSSILinkType(str, Enum):
    RSSI_LOS = "RSSI_LOS"
    RSSI_NLOS = "RSSI_NLOS"



"""------------------------------------------------------------------------------
 🎛️ CHANNELS 📶
------------------------------------------------------------------------------"""

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


def humans_blocking_count(
    a: Tuple[float, float],
    b: Tuple[float, float],
    humans: List[Human],
) -> int:
    count = 0
    for human in humans:
        if human.radius is not None:
            if _segment_intersects_circle(a, b, human.position_X_Y, human.radius):
                count += 1
        else:
            if human.position_X1_Y1 is None:
                continue
            x0, y0 = human.position_X_Y
            x1, y1 = human.position_X1_Y1
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
    blocking_humans: int


class ChannelFactory:
    """
    Builds channels for a scenario.

    NOTE:
    - INDOOR_LOS = minimum of 1 completely free channel
    - INDOOR_NLOS = not allowed to have any free channels (all must have at least 1 human)
    """
    def __init__(
        self,
        env: Environment,
        humans: List[Human],
    ) -> None:
        self.env = env
        self.humans = humans

    def _make_channel(self, a: Device, b: Device) -> Channel:
        d = euclidean_distance(a.position, b.position)
        blocks = humans_blocking_count(a.position, b.position, self.humans)

        return Channel(
            device_a_id=a.device_id,
            device_a_position=a.position,
            device_b_id=b.device_id,
            device_b_position=b.position,
            distance_m=d,
            blocking_humans=blocks,
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
    - OUTDOOR: at least 80% of channels must have 0 blocking humans (completely free).
    - INDOOR_LOS: at least one channel must have 0 blocking humans (completely free).
    - INDOOR_NLOS: no channel may be completely free (each must have >=1 blocker).
    """
    if not channels:
        return True  # nothing to validate

    if env_type == EnvironmentType.OUTDOOR:
        free = sum(1 for ch in channels if ch.blocking_humans == 0)
        return free / len(channels) >= 0.80

    if env_type == EnvironmentType.INDOOR_LOS:
        return any(ch.blocking_humans == 0 for ch in channels)

    if env_type == EnvironmentType.INDOOR_NLOS:
        return all(ch.blocking_humans >= 1 for ch in channels)

    return True


def _device_pairs_for_channels(devices: List[Device]) -> List[Tuple[Device, Device]]:
    antennas = [d for d in devices if d.device_type == DeviceType.ANTENNA]
    targets = [d for d in devices if d.device_type == DeviceType.TARGET]
    pairs: List[Tuple[Device, Device]] = []
    if antennas and targets:
        for a in antennas:
            for t in targets:
                pairs.append((a, t))
    else:
        for i in range(len(devices)):
            for j in range(i + 1, len(devices)):
                pairs.append((devices[i], devices[j]))
    return pairs

# This is probably not the most efficient way of doing this but it id good enough for now
# Could be optimised and improved in the future
def remove_blocking_humans_for_outdoor(
    env_type: EnvironmentType, humans: List[Human], devices: List[Device], *, target_free_ratio: float = 0.80
) -> List[Human]:
    """
    For OUTDOOR envs, greedily remove humans that block the most pairs until the fraction of free channels reaches the target ratio.
    """
    if env_type != EnvironmentType.OUTDOOR or not humans or len(devices) < 2:
        return humans

    pairs = _device_pairs_for_channels(devices)
    kept = list(humans)

    def free_ratio(current_humans: List[Human]) -> float:
        if not pairs:
            return 1.0
        blocked = 0
        for a, b in pairs:
            if humans_blocking_count(a.position, b.position, current_humans) > 0:
                blocked += 1
        return 1 - (blocked / len(pairs))

    # Remove worst-offending human until we meet target or nothing blocks
    for _ in range(len(humans)):
        if free_ratio(kept) >= target_free_ratio:
            break

        # score humans by how many pairs they alone block
        scores = []
        for human in kept:
            blocked_pairs = 0
            for a, b in pairs:
                if humans_blocking_count(a.position, b.position, [human]) > 0:
                    blocked_pairs += 1
            scores.append((blocked_pairs, human))

        # pick human that blocks the most pairs; if none block, stop
        scores.sort(key=lambda t: t[0], reverse=True)
        if scores and scores[0][0] > 0:
            kept.remove(scores[0][1])
        else:
            break

    return kept


def remove_blocking_humans_for_indoor_los(
    env_type: EnvironmentType, humans: List[Human], devices: List[Device]
) -> List[Human]:
    """
    For INDOOR_LOS envs, ensure at least one channel can be completely free by
    removing the minimum human blockers for a device pair.
    """
    if env_type != EnvironmentType.INDOOR_LOS or not humans or len(devices) < 2:
        return humans

    pairs = _device_pairs_for_channels(devices)
    if not pairs:
        return humans

    kept = list(humans)

    # If already valid, keep as-is.
    for a, b in pairs:
        if humans_blocking_count(a.position, b.position, kept) == 0:
            return kept

    # Choose the pair with the fewest blockers and remove only those blockers.
    best_pair: Optional[Tuple[Device, Device]] = None
    best_blockers: Optional[List[Human]] = None

    for a, b in pairs:
        blockers: List[Human] = []
        for human in kept:
            if humans_blocking_count(a.position, b.position, [human]) > 0:
                blockers.append(human)

        if best_blockers is None or len(blockers) < len(best_blockers):
            best_pair = (a, b)
            best_blockers = blockers

            if len(best_blockers) == 0:
                break

    if best_pair is None or best_blockers is None:
        return kept

    for ob in best_blockers:
        if ob in kept:
            kept.remove(ob)

    return kept
