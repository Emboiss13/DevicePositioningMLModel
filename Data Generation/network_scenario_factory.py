from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional
import random
from antenna_factory import Antenna, AntennaFactory
from obstacle_factory import Obstacle, ObstacleFactory
from environment_factory import Environment


"""

🛜 NETWORK ENVIRONMENT SCENARIO FACTORY 📍
------------------------------------------

This module defines the core data structures and random generation logic for our network scenarios, including:

- Environment
- Antennas
- Obstacles

"""



@dataclass
class NetworkScenario:
    environment: Environment
    antennas: List[Antenna]                   # E.g. {antenna_1: {position: [3,4], radius: 3, covered_space: {total_area: 7, covered_coordinates: [[0,6], [1,7]]}}}
    obstacles: List[Obstacle]                 # E.g. {obstacle_1: {position: [3,4], hight: 2, length: 45, covered_space: {total_area: 7, covered_coordinates: [[0,6], [1,7]]}}}
    #targets: List[Targets]                 

    @classmethod
    def generate_random(
        cls,
        *,
        label: str = "random_scenario",
        seed: Optional[int] = None,
    ) -> "NetworkScenario":
        if seed is not None:
            random.seed(seed)

        env = Environment()

        # ---- Antennas
        antenna_factory = AntennaFactory(env)
        antennas: List[Antenna] = []

        while antenna_factory.has_capacity():
            antennas.append(antenna_factory.create_antenna())

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

        return cls(
            environment=env,
            antennas=antennas,
            obstacles=obstacles,
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        JSON-style formatting for generated data.
        """
        env = {
            "width": self.environment.width,
            "height": self.environment.height,
            "grid_area": self.environment.grid_area,
            "x_domain": self.environment.x_domain,
            "y_range": self.environment.y_range,
        }

        antennas = []
        for antenna in self.antennas:
            antennas.append(asdict(antenna))

        obstacles = []
        for obstacle in self.obstacles:
            obstacle_dict = asdict(obstacle)
            obstacle_dict["obstacle_type"] = obstacle.obstacle_type.value
            obstacles.append(obstacle_dict)

        return {
            "environment": env,
            "antennas": antennas,
            "obstacles": obstacles,
        }
