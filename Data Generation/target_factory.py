from dataclasses import dataclass
from typing import Tuple

@dataclass
class Target:
    target_id: int
    target_label: str
    position: Tuple[float, float]