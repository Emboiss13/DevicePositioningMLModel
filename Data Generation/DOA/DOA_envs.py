"""
DOA ENVIRONMENT LAYER
---------------------

This module derives DOA/AOA-specific columns from the shared per-link geometry
table produced by `link_factory.py`.

Input tables:
- env_summary.parquet
- links.parquet

Output table:
- links_doa.parquet by default

The resulting rows remain one row per antenna-target link. Full MUSIC snapshot
simulation is intentionally deferred; this layer generates geometric bearings
with environment/link-state angular noise for downstream positioning and ML.

Usage examples:
python3 "Data Generation/DOA/DOA_envs.py" --data-dir "generated_network_scenarios"
python3 "Data Generation/DOA/DOA_envs.py" --data-dir "generated_network_scenarios" --seed 42
python3 "Data Generation/DOA/DOA_envs.py" \
  --data-dir "Data Generation/generated_network_scenarios" \
  --carrier-frequency-hz 5500000000.0
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd


DEFAULT_CARRIER_FREQUENCY_HZ = 5_500_000_000.0
DEFAULT_PROPAGATION_SPEED_M_PER_S = 299_792_458.0
DEFAULT_NUM_ARRAY_ELEMENTS_MIN = 5
DEFAULT_NUM_ARRAY_ELEMENTS_MAX = 10
DEFAULT_ARRAY_ORIENTATION_RAD = 0.0
DOA_ZERO_DISTANCE_EPSILON_M = 1e-9

DOA_NOISE_SIGMA_DEG_RANGES = {
    "outdoor_los": (1.0, 3.0),
    "outdoor_nlos": (3.0, 8.0),
    "indoor_los": (2.0, 5.0),
    "indoor_nlos": (5.0, 15.0),
}

REQUIRED_LINK_COLUMNS = {
    "scenario_id",
    "target_id",
    "target_label",
    "target_x",
    "target_y",
    "antenna_id",
    "antenna_label",
    "antenna_x",
    "antenna_y",
    "distance_m",
    "link_state",
}

NoiseRangeMap = Dict[str, Tuple[float, float]]


def _wrap_angle_rad(angle):
    """Wrap radians to [-pi, pi). Works for scalars and numpy/pandas arrays."""
    return ((angle + math.pi) % (2.0 * math.pi)) - math.pi


def _wrap_angle_deg(angle):
    """Wrap degrees to [-180, 180). Works for scalars and numpy/pandas arrays."""
    return ((angle + 180.0) % 360.0) - 180.0


def _normalize_env_type(env: str) -> str:
    normalized = str(env).strip().lower()
    if normalized not in {"indoor", "outdoor"}:
        raise ValueError(f"Unsupported DOA environment type: {env}")
    return normalized


def _normalize_link_state(link_state: str) -> str:
    normalized = str(link_state).strip().upper()
    if normalized not in {"LOS", "NLOS"}:
        raise ValueError(f"Unsupported DOA link_state: {link_state}")
    return normalized


def _resolve_doa_env_type(env_type: str, link_state: str) -> str:
    env = _normalize_env_type(env_type)
    state = _normalize_link_state(link_state).lower()
    return f"{env}_{state}"


def _validate_noise_range(name: str, lo: float, hi: float) -> None:
    if lo < 0 or hi < 0:
        raise ValueError(f"{name} noise sigma range values must be >= 0.")
    if lo > hi:
        raise ValueError(f"{name} noise sigma range minimum must be <= maximum.")


def _validate_noise_ranges(noise_ranges: NoiseRangeMap) -> None:
    for env in DOA_NOISE_SIGMA_DEG_RANGES:
        if env not in noise_ranges:
            raise ValueError(f"Missing DOA noise sigma range for environment: {env}")
        lo, hi = noise_ranges[env]
        _validate_noise_range(env, float(lo), float(hi))


def _validate_array_element_range(
    num_array_elements_min: int,
    num_array_elements_max: int,
) -> None:
    if num_array_elements_min < 1 or num_array_elements_max < 1:
        raise ValueError("num_array_elements_min and num_array_elements_max must be >= 1.")
    if num_array_elements_min > num_array_elements_max:
        raise ValueError("num_array_elements_min must be <= num_array_elements_max.")


def angular_noise_sigma_given_env_type(
    env: str,
    *,
    noise_ranges: Optional[NoiseRangeMap] = None,
    rng: Optional[random.Random] = None,
) -> float:
    normalized = str(env).strip().lower()
    ranges = noise_ranges or DOA_NOISE_SIGMA_DEG_RANGES
    if normalized not in ranges:
        raise ValueError(f"Unsupported DOA environment type: {env}")

    lo, hi = ranges[normalized]
    sampler = rng or random
    return sampler.uniform(float(lo), float(hi))


def extract_doa_link_inputs(data_dir: Union[str, Path]) -> pd.DataFrame:
    """
    Return one row per antenna-target link with the scenario environment joined in.

    Required columns:
    - scenario_id
    - target_id
    - target_label
    - target_x
    - target_y
    - antenna_id
    - antenna_label
    - antenna_x
    - antenna_y
    - distance_m
    - link_state
    """
    data_path = Path(data_dir)
    summary_path = data_path / "env_summary.parquet"
    links_path = data_path / "links.parquet"

    summary_env_df = pd.read_parquet(summary_path, columns=["scenario_id", "env_type"])
    links_df = pd.read_parquet(links_path)

    missing_cols = REQUIRED_LINK_COLUMNS.difference(links_df.columns)
    if missing_cols:
        missing_list = ", ".join(sorted(missing_cols))
        raise ValueError(f"links.parquet is missing required columns: {missing_list}")

    if "env_type" in links_df.columns:
        links_df = links_df.drop(columns=["env_type"])

    merged_df = links_df.merge(
        summary_env_df,
        on="scenario_id",
        how="left",
        validate="many_to_one",
    )
    if merged_df["env_type"].isna().any():
        missing_count = int(merged_df["env_type"].isna().sum())
        raise ValueError(
            f"{missing_count} link rows do not map to a scenario environment."
        )

    merged_df["env_type"] = merged_df["env_type"].map(_normalize_env_type)
    merged_df["doa_env_type"] = [
        _resolve_doa_env_type(env_type, link_state)
        for env_type, link_state in zip(merged_df["env_type"], merged_df["link_state"])
    ]
    return merged_df.copy()


def create_scenario_doa_parameters(
    link_inputs_df: pd.DataFrame,
    *,
    seed: Optional[int] = None,
    carrier_frequency_hz: float = DEFAULT_CARRIER_FREQUENCY_HZ,
    propagation_speed_m_per_s: float = DEFAULT_PROPAGATION_SPEED_M_PER_S,
    num_array_elements_min: int = DEFAULT_NUM_ARRAY_ELEMENTS_MIN,
    num_array_elements_max: int = DEFAULT_NUM_ARRAY_ELEMENTS_MAX,
    element_spacing_m: Optional[float] = None,
    array_orientation_rad: float = DEFAULT_ARRAY_ORIENTATION_RAD,
    noise_ranges: Optional[NoiseRangeMap] = None,
) -> pd.DataFrame:
    """
    Sample DOA parameters once per scenario and per derived DOA environment class.

    The ULA orientation is fixed by default to global +x broadside. The number of
    array elements is sampled once per scenario, while angular-noise sigma is
    sampled once per scenario and DOA environment class.
    """
    if carrier_frequency_hz <= 0:
        raise ValueError("carrier_frequency_hz must be > 0.")
    if propagation_speed_m_per_s <= 0:
        raise ValueError("propagation_speed_m_per_s must be > 0.")
    if not math.isfinite(array_orientation_rad):
        raise ValueError("array_orientation_rad must be finite.")

    _validate_array_element_range(num_array_elements_min, num_array_elements_max)

    ranges = noise_ranges or DOA_NOISE_SIGMA_DEG_RANGES
    _validate_noise_ranges(ranges)

    wavelength_m = float(propagation_speed_m_per_s) / float(carrier_frequency_hz)
    resolved_element_spacing_m = (
        wavelength_m / 2.0 if element_spacing_m is None else float(element_spacing_m)
    )
    if resolved_element_spacing_m <= 0:
        raise ValueError("element_spacing_m must be > 0.")
    if resolved_element_spacing_m > wavelength_m / 2.0:
        raise ValueError("element_spacing_m must be <= wavelength_m / 2 to avoid spatial aliasing.")

    wrapped_orientation_rad = float(_wrap_angle_rad(float(array_orientation_rad)))
    wrapped_orientation_deg = float(_wrap_angle_deg(math.degrees(wrapped_orientation_rad)))

    rng = random.Random(seed)
    rows = []
    for scenario_id, scenario_df in link_inputs_df.groupby("scenario_id", sort=True):
        env_values = scenario_df["env_type"].drop_duplicates().tolist()
        if len(env_values) != 1:
            raise ValueError(
                f"Scenario {scenario_id} maps to {len(env_values)} environment types."
            )

        env_type = _normalize_env_type(env_values[0])
        num_array_elements = rng.randint(
            int(num_array_elements_min),
            int(num_array_elements_max),
        )
        doa_env_types = sorted(scenario_df["doa_env_type"].drop_duplicates().tolist())
        for doa_env_type in doa_env_types:
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "env_type": env_type,
                    "doa_env_type": doa_env_type,
                    "carrier_frequency_hz": float(carrier_frequency_hz),
                    "propagation_speed_m_per_s": float(propagation_speed_m_per_s),
                    "wavelength_m": wavelength_m,
                    "num_array_elements": num_array_elements,
                    "element_spacing_m": resolved_element_spacing_m,
                    "array_orientation_rad": wrapped_orientation_rad,
                    "array_orientation_deg": wrapped_orientation_deg,
                    "doa_noise_sigma_deg": angular_noise_sigma_given_env_type(
                        doa_env_type,
                        noise_ranges=ranges,
                        rng=rng,
                    ),
                }
            )

    scenario_doa_df = pd.DataFrame(rows)
    if scenario_doa_df.empty:
        return pd.DataFrame(
            columns=[
                "scenario_id",
                "env_type",
                "doa_env_type",
                "carrier_frequency_hz",
                "propagation_speed_m_per_s",
                "wavelength_m",
                "num_array_elements",
                "element_spacing_m",
                "array_orientation_rad",
                "array_orientation_deg",
                "doa_noise_sigma_deg",
                "doa_noise_sigma_rad",
            ]
        )

    scenario_doa_df["doa_noise_sigma_rad"] = np.deg2rad(
        scenario_doa_df["doa_noise_sigma_deg"].astype(float)
    )
    return scenario_doa_df


def build_doa_base_table(
    data_dir: Union[str, Path],
    *,
    seed: Optional[int] = None,
    carrier_frequency_hz: float = DEFAULT_CARRIER_FREQUENCY_HZ,
    propagation_speed_m_per_s: float = DEFAULT_PROPAGATION_SPEED_M_PER_S,
    num_array_elements_min: int = DEFAULT_NUM_ARRAY_ELEMENTS_MIN,
    num_array_elements_max: int = DEFAULT_NUM_ARRAY_ELEMENTS_MAX,
    element_spacing_m: Optional[float] = None,
    outdoor_los_noise_sigma_deg_min: float = 1.0,
    outdoor_los_noise_sigma_deg_max: float = 3.0,
    outdoor_nlos_noise_sigma_deg_min: float = 3.0,
    outdoor_nlos_noise_sigma_deg_max: float = 8.0,
    indoor_los_noise_sigma_deg_min: float = 2.0,
    indoor_los_noise_sigma_deg_max: float = 5.0,
    indoor_nlos_noise_sigma_deg_min: float = 5.0,
    indoor_nlos_noise_sigma_deg_max: float = 15.0,
) -> pd.DataFrame:
    """
    Return the shared link table with DOA/AOA columns added.

    Added columns:
    - doa_env_type
    - carrier_frequency_hz
    - propagation_speed_m_per_s
    - wavelength_m
    - num_array_elements
    - element_spacing_m
    - array_orientation_rad
    - array_orientation_deg
    - true_bearing_rad
    - true_bearing_deg
    - true_doa_rad
    - true_doa_deg
    - doa_noise_sigma_rad
    - doa_noise_sigma_deg
    - doa_noise_rad
    - doa_noise_deg
    - observed_doa_rad
    - observed_doa_deg
    - observed_bearing_rad
    - observed_bearing_deg
    - is_doa_valid
    """
    noise_ranges = {
        "outdoor_los": (
            float(outdoor_los_noise_sigma_deg_min),
            float(outdoor_los_noise_sigma_deg_max),
        ),
        "outdoor_nlos": (
            float(outdoor_nlos_noise_sigma_deg_min),
            float(outdoor_nlos_noise_sigma_deg_max),
        ),
        "indoor_los": (
            float(indoor_los_noise_sigma_deg_min),
            float(indoor_los_noise_sigma_deg_max),
        ),
        "indoor_nlos": (
            float(indoor_nlos_noise_sigma_deg_min),
            float(indoor_nlos_noise_sigma_deg_max),
        ),
    }
    _validate_noise_ranges(noise_ranges)

    links_df = extract_doa_link_inputs(data_dir)

    if (links_df["distance_m"] < 0).any():
        bad_count = int((links_df["distance_m"] < 0).sum())
        raise ValueError(
            f"distance_m must be >= 0 for DOA calculations. Found {bad_count} invalid rows."
        )
    duplicated_links = links_df.duplicated(
        ["scenario_id", "target_id", "antenna_id"],
        keep=False,
    )
    if duplicated_links.any():
        duplicate_count = int(duplicated_links.sum())
        raise ValueError(
            f"Found {duplicate_count} duplicate links by scenario_id, target_id, and antenna_id."
        )

    scenario_params_df = create_scenario_doa_parameters(
        links_df,
        seed=seed,
        carrier_frequency_hz=carrier_frequency_hz,
        propagation_speed_m_per_s=propagation_speed_m_per_s,
        num_array_elements_min=num_array_elements_min,
        num_array_elements_max=num_array_elements_max,
        element_spacing_m=element_spacing_m,
        array_orientation_rad=DEFAULT_ARRAY_ORIENTATION_RAD,
        noise_ranges=noise_ranges,
    )

    doa_df = links_df.merge(
        scenario_params_df,
        on=["scenario_id", "env_type", "doa_env_type"],
        how="left",
        validate="many_to_one",
    )

    if doa_df["carrier_frequency_hz"].isna().any():
        missing_count = int(doa_df["carrier_frequency_hz"].isna().sum())
        raise ValueError(f"{missing_count} DOA rows do not have scenario parameters.")

    np_rng = np.random.default_rng(seed)
    doa_df["is_doa_valid"] = (
        doa_df["distance_m"].astype(float) > DOA_ZERO_DISTANCE_EPSILON_M
    )

    angle_cols = [
        "true_bearing_rad",
        "true_bearing_deg",
        "true_doa_rad",
        "true_doa_deg",
        "doa_noise_rad",
        "doa_noise_deg",
        "observed_doa_rad",
        "observed_doa_deg",
        "observed_bearing_rad",
        "observed_bearing_deg",
    ]
    for column in angle_cols:
        doa_df[column] = np.nan

    valid_mask = doa_df["is_doa_valid"]
    if valid_mask.any():
        valid_idx = doa_df.index[valid_mask]
        dx = (
            doa_df.loc[valid_idx, "target_x"].astype(float)
            - doa_df.loc[valid_idx, "antenna_x"].astype(float)
        )
        dy = (
            doa_df.loc[valid_idx, "target_y"].astype(float)
            - doa_df.loc[valid_idx, "antenna_y"].astype(float)
        )

        true_bearing_rad = _wrap_angle_rad(np.arctan2(dy, dx))
        array_orientation_rad = doa_df.loc[valid_idx, "array_orientation_rad"].astype(float)
        true_doa_rad = _wrap_angle_rad(true_bearing_rad - array_orientation_rad)

        doa_noise_rad = _wrap_angle_rad(
            np_rng.normal(
                loc=0.0,
                scale=doa_df.loc[valid_idx, "doa_noise_sigma_rad"].astype(float).to_numpy(),
                size=len(valid_idx),
            )
        )
        observed_doa_rad = _wrap_angle_rad(true_doa_rad + doa_noise_rad)
        observed_bearing_rad = _wrap_angle_rad(observed_doa_rad + array_orientation_rad)

        doa_df.loc[valid_idx, "true_bearing_rad"] = true_bearing_rad
        doa_df.loc[valid_idx, "true_bearing_deg"] = _wrap_angle_deg(
            np.degrees(true_bearing_rad)
        )
        doa_df.loc[valid_idx, "true_doa_rad"] = true_doa_rad
        doa_df.loc[valid_idx, "true_doa_deg"] = _wrap_angle_deg(np.degrees(true_doa_rad))
        doa_df.loc[valid_idx, "doa_noise_rad"] = doa_noise_rad
        doa_df.loc[valid_idx, "doa_noise_deg"] = _wrap_angle_deg(np.degrees(doa_noise_rad))
        doa_df.loc[valid_idx, "observed_doa_rad"] = observed_doa_rad
        doa_df.loc[valid_idx, "observed_doa_deg"] = _wrap_angle_deg(
            np.degrees(observed_doa_rad)
        )
        doa_df.loc[valid_idx, "observed_bearing_rad"] = observed_bearing_rad
        doa_df.loc[valid_idx, "observed_bearing_deg"] = _wrap_angle_deg(
            np.degrees(observed_bearing_rad)
        )

    added_cols = [
        "carrier_frequency_hz",
        "propagation_speed_m_per_s",
        "wavelength_m",
        "num_array_elements",
        "element_spacing_m",
        "array_orientation_rad",
        "array_orientation_deg",
        "true_bearing_rad",
        "true_bearing_deg",
        "true_doa_rad",
        "true_doa_deg",
        "doa_noise_sigma_rad",
        "doa_noise_sigma_deg",
        "doa_noise_rad",
        "doa_noise_deg",
        "observed_doa_rad",
        "observed_doa_deg",
        "observed_bearing_rad",
        "observed_bearing_deg",
        "is_doa_valid",
    ]
    ordered_cols = [
        *links_df.columns,
        *[column for column in added_cols if column not in links_df.columns],
    ]
    return doa_df[ordered_cols].copy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build DOA/AOA angle columns from env_summary.parquet and links.parquet."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="generated_network_scenarios",
        help="Directory containing env_summary.parquet and links.parquet.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed for reproducible DOA parameter and angular-noise sampling.",
    )
    parser.add_argument(
        "--carrier-frequency-hz",
        type=float,
        default=DEFAULT_CARRIER_FREQUENCY_HZ,
        help="Carrier frequency in Hz used to compute wavelength.",
    )
    parser.add_argument(
        "--propagation-speed-m-per-s",
        type=float,
        default=DEFAULT_PROPAGATION_SPEED_M_PER_S,
        help="Signal propagation speed in meters per second.",
    )
    parser.add_argument(
        "--num-array-elements-min",
        type=int,
        default=DEFAULT_NUM_ARRAY_ELEMENTS_MIN,
        help="Minimum sampled ULA element count per scenario.",
    )
    parser.add_argument(
        "--num-array-elements-max",
        type=int,
        default=DEFAULT_NUM_ARRAY_ELEMENTS_MAX,
        help="Maximum sampled ULA element count per scenario.",
    )
    parser.add_argument(
        "--element-spacing-m",
        type=float,
        default=None,
        help="Optional ULA element spacing in meters. Default: wavelength / 2.",
    )
    parser.add_argument(
        "--outdoor-los-noise-sigma-deg-min",
        type=float,
        default=1.0,
        help="Minimum outdoor LOS angular-noise sigma in degrees.",
    )
    parser.add_argument(
        "--outdoor-los-noise-sigma-deg-max",
        type=float,
        default=3.0,
        help="Maximum outdoor LOS angular-noise sigma in degrees.",
    )
    parser.add_argument(
        "--outdoor-nlos-noise-sigma-deg-min",
        type=float,
        default=3.0,
        help="Minimum outdoor NLOS angular-noise sigma in degrees.",
    )
    parser.add_argument(
        "--outdoor-nlos-noise-sigma-deg-max",
        type=float,
        default=8.0,
        help="Maximum outdoor NLOS angular-noise sigma in degrees.",
    )
    parser.add_argument(
        "--indoor-los-noise-sigma-deg-min",
        type=float,
        default=2.0,
        help="Minimum indoor LOS angular-noise sigma in degrees.",
    )
    parser.add_argument(
        "--indoor-los-noise-sigma-deg-max",
        type=float,
        default=5.0,
        help="Maximum indoor LOS angular-noise sigma in degrees.",
    )
    parser.add_argument(
        "--indoor-nlos-noise-sigma-deg-min",
        type=float,
        default=5.0,
        help="Minimum indoor NLOS angular-noise sigma in degrees.",
    )
    parser.add_argument(
        "--indoor-nlos-noise-sigma-deg-max",
        type=float,
        default=15.0,
        help="Maximum indoor NLOS angular-noise sigma in degrees.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output parquet path. Default: <data-dir>/links_doa.parquet",
    )
    parser.add_argument(
        "--overwrite-links",
        action="store_true",
        help="If set, write output directly to <data-dir>/links.parquet.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if args.overwrite_links:
        output_path = data_dir / "links.parquet"
    elif args.output:
        output_path = Path(args.output)
    else:
        output_path = data_dir / "links_doa.parquet"

    doa_df = build_doa_base_table(
        data_dir=data_dir,
        seed=args.seed,
        carrier_frequency_hz=args.carrier_frequency_hz,
        propagation_speed_m_per_s=args.propagation_speed_m_per_s,
        num_array_elements_min=args.num_array_elements_min,
        num_array_elements_max=args.num_array_elements_max,
        element_spacing_m=args.element_spacing_m,
        outdoor_los_noise_sigma_deg_min=args.outdoor_los_noise_sigma_deg_min,
        outdoor_los_noise_sigma_deg_max=args.outdoor_los_noise_sigma_deg_max,
        outdoor_nlos_noise_sigma_deg_min=args.outdoor_nlos_noise_sigma_deg_min,
        outdoor_nlos_noise_sigma_deg_max=args.outdoor_nlos_noise_sigma_deg_max,
        indoor_los_noise_sigma_deg_min=args.indoor_los_noise_sigma_deg_min,
        indoor_los_noise_sigma_deg_max=args.indoor_los_noise_sigma_deg_max,
        indoor_nlos_noise_sigma_deg_min=args.indoor_nlos_noise_sigma_deg_min,
        indoor_nlos_noise_sigma_deg_max=args.indoor_nlos_noise_sigma_deg_max,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doa_df.to_parquet(output_path, index=False)

    print(f"Wrote {len(doa_df)} DOA link rows to {output_path}")
    if args.seed is None:
        print("Seed: none (sampling varies across runs).")
    else:
        print(f"Seed: {args.seed} (reproducible sampling).")


if __name__ == "__main__":
    main()
