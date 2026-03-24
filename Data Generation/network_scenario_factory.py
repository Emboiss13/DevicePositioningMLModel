"""
NETWORK ENVIRONMENT SCENARIO FACTORY
------------------------------------

This module defines the core data structures and random generation logic for our network scenarios, including:

- Environment
- Floorplan
- Antennas
- Humans

@author: Giuliana Emberson
@date: 7th of May 2026

"""


from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional
import random
from antenna_factory import Antenna, AntennaFactory
from environment_factory import Environment
from human_factory import Human, HumanFactory


@dataclass
class NetworkScenario:
    environment: Environment
    antennas: List[Antenna]                   # E.g. {antenna_1: {position: [3,4], radius: 3, covered_space: {total_area: 7, covered_coordinates: [[0,6], [1,7]]}}}
    humans: List[Human]                       # E.g. {human_1: {position: [3,4], height: 2, length: 45, covered_space: {total_area: 7, covered_coordinates: [[0,6], [1,7]]}}}
    #targets: List[Targets]                 

    @classmethod
    def generate_random(
        cls,
        *,
        seed: Optional[int] = None,
    ) -> "NetworkScenario":
        if seed is not None:
            random.seed(seed)

        env = Environment()

        # Antennas
        antenna_factory = AntennaFactory(env)
        antennas: List[Antenna] = []

        while antenna_factory.has_capacity():
            antennas.append(antenna_factory.create_antenna())

        # Humans
        human_factory = HumanFactory(env)
        humans: List[Human] = []
        failed_draws = 0
        max_failed_draws = 50  # safety only

        while human_factory.can_fit_any_human():
            try:
                human = human_factory.create_human()
                humans.append(human)
                failed_draws = 0  # reset after success
            except RuntimeError:
                failed_draws += 1
                if failed_draws >= max_failed_draws:
                    break

        return cls(
            environment=env,
            antennas=antennas,
            humans=humans,
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        JSON-style formatting for generated data
        """
        env = {
            "width": self.environment.width,
            "height": self.environment.height,
            "area": self.environment.area,
            "env_type": self.environment.env_type,
            "x_domain": self.environment.x_domain,
            "y_range": self.environment.y_range,
        }

        antennas = []
        for antenna in self.antennas:
            antennas.append(asdict(antenna))

        humans = []
        for human in self.humans:
            human_dict = asdict(human)
            humans.append(human_dict)

        return {
            "environment": env,
            "antennas": antennas,
            "humans": humans,
        }
