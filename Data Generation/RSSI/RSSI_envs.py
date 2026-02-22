from general_envs import EnvironmentType
from __future__ import annotations
from dataclasses import asdict, dataclass, field
from enum import Enum
import math
from typing import Any, Dict, List, Tuple, Optional
import random


"""

RSSI SPECIFIC ATTRIBUTES AND METHODS
------------------------------------

This module defines the core data structures and random generation logic for RSSI to calculate a position estimation 
given a given network scenarios, including:

    • Path Loss exponent: The average rate at which signal power decays with distance and depends on the propagation environment.

    • Log-Distance Path Loss Model: The large-scale average path loss between two devices separated by distance (d) 
      is modelled using the log-distance path loss model

    • Log-Normal Shadowing (Gaussian Noise): To model environmental variability and measurement uncertainty, log-normal shadowing 
      is applied by adding a zero-mean Gaussian random variable Xσ in the logarithmic domain

  
"""

def path_loss_exponent_given_env_type(env: EnvironmentType) -> float:
  if (env == "outdoor") :
    return random.uniform(2.7, 3.5)
  if (env == "indoor_LOS") :
    return random.uniform(1.6, 1.8)
  if (env == "indoor_NLOS") :
    return random.uniform(4.0, 6.0)
  


