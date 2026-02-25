from __future__ import annotations
import argparse
import random
from pathlib import Path
from typing import Union
import numpy as np
import pandas as pd
from general_envs_factory import EnvironmentType

# NOTE: df = data frame

"""
RSSI SPECIFIC ATTRIBUTES AND METHODS
------------------------------------

This module defines the core data structures and random generation logic for RSSI to calculate a position estimation
given a given network scenarios, using equation:

    RSSI = -10nlog_{10}(d)+A

        n) Path-loss exponent, which indicates the rate at which signal strength decreases relative to distance (
        d) The distance between the transmitter and the receiver (typically in meters).
        A) The reference value measured at a distance of 1 meter from the transmitter. 

    - Path Loss exponent: The average rate at which signal power decays with distance and depends on the propagation environment.

    - Log-Distance Path Loss Model: The large-scale average path loss between two devices separated by distance (d)
      is modelled using the log-distance path loss model.

    - Log-Normal Shadowing (Gaussian Noise): To model environmental variability and measurement uncertainty, log-normal shadowing
      is applied by adding a zero-mean Gaussian random variable in the logarithmic domain.


RUNNIG SCRIPT
-------------
python3 "Data Generation/RSSI/RSSI_envs.py" --data-dir "generated_network_scenarios"
python3 "Data Generation/RSSI/RSSI_envs.py" --data-dir "generated_network_scenarios" --seed 42 --reference-distance-m 1.0



"""


PATH_LOSS_EXPONENT_RANGES = {
    EnvironmentType.OUTDOOR: (2.7, 3.5),
    EnvironmentType.INDOOR_LOS: (1.6, 1.8),
    EnvironmentType.INDOOR_NLOS: (4.0, 6.0),
}

SHADOW_SIGMA_DB_RANGES = {
    EnvironmentType.OUTDOOR: (4, 12),
    EnvironmentType.INDOOR_LOS: (5, 12),
    EnvironmentType.INDOOR_NLOS: (5, 12),
}

A_DBM_RANGES = {
    EnvironmentType.OUTDOOR: (-55.0, -35.0),
    EnvironmentType.INDOOR_LOS: (-50.0, -30.0),
    EnvironmentType.INDOOR_NLOS: (-65.0, -40.0),
}


def _as_env_type(env: Union[str, EnvironmentType]) -> EnvironmentType:
    if isinstance(env, EnvironmentType):
        return env
    return EnvironmentType(env)


def path_loss_exponent_given_env_type(env: Union[str, EnvironmentType]) -> float:
    env_type = _as_env_type(env)
    lo, hi = PATH_LOSS_EXPONENT_RANGES[env_type]
    return random.uniform(lo, hi)


def shadow_sigma_given_env_type(env: Union[str, EnvironmentType]) -> float:
    env_type = _as_env_type(env)
    lo, hi = SHADOW_SIGMA_DB_RANGES[env_type]
    return random.uniform(lo, hi)


def extract_rssi_channel_inputs(data_dir: Union[str, Path]) -> pd.DataFrame:
    """
    Return one row per channel with channel columns preserved plus environment label.
    Required columns for RSSI formulas:
    - scenario_id
    - env_type
    - distance_m
    """
    data_path = Path(data_dir)
    summary_path = data_path / "summary.parquet"
    channels_path = data_path / "channels.parquet"

    summary_env_df = pd.read_parquet(summary_path, columns=["scenario_id", "env_type"])
    channels_df = pd.read_parquet(channels_path)

    required_channels_cols = {"scenario_id", "distance_m"}
    missing_cols = required_channels_cols.difference(channels_df.columns)
    if missing_cols:
        missing_list = ", ".join(sorted(missing_cols))
        raise ValueError(f"channels.parquet is missing required columns: {missing_list}")

    if "env_type" in channels_df.columns:
        channels_df = channels_df.drop(columns=["env_type"])

    merged_extracted_data = channels_df.merge(
        summary_env_df,
        on="scenario_id",
        how="left",
        validate="many_to_one",
    )
    if merged_extracted_data["env_type"].isna().any():
        missing_count = int(merged_extracted_data["env_type"].isna().sum())
        raise ValueError(
            f"{missing_count} channel rows do not map to an environment (missing scenario_id in summary)."
        )

    return merged_extracted_data.copy()


def create_scenario_rssi_parameters(
    link_inputs_df: pd.DataFrame,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Sample scenario-level RSSI parameters from env_type ranges.
    """
    rng = random.Random(seed)
    rssi_network_scenario = link_inputs_df[["scenario_id", "env_type"]].drop_duplicates().copy()

    def sample_n(env: str) -> float:
        env_type = _as_env_type(env)
        lo, hi = PATH_LOSS_EXPONENT_RANGES[env_type]
        return rng.uniform(lo, hi)

    def sample_sigma(env: str) -> float:
        env_type = _as_env_type(env)
        lo, hi = SHADOW_SIGMA_DB_RANGES[env_type]
        return rng.uniform(lo, hi)

    def sample_a_dbm(env: str) -> float:
        env_type = _as_env_type(env)
        lo, hi = A_DBM_RANGES[env_type]
        return rng.uniform(lo, hi)

    rssi_network_scenario["path_loss_exponent_n"] = rssi_network_scenario["env_type"].map(sample_n)
    rssi_network_scenario["shadow_sigma_db"] = rssi_network_scenario["env_type"].map(sample_sigma)
    rssi_network_scenario["initial_signal_strength_dbm"] = rssi_network_scenario["env_type"].map(sample_a_dbm)
    return rssi_network_scenario


def build_rssi_base_table(
    data_dir: Union[str, Path],
    seed: int | None = None,
    reference_a_dbm: float | None = None,
    reference_distance_m: float = 1.0,
) -> pd.DataFrame:
    """
    Return per-channel table with original channel fields plus RSSI/path-loss attributes.

    Added columns:
    - path_loss_exponent_n (scenario-level)
    - shadow_sigma_db (scenario-level)
    - initial_signal_strength_dbm (A at reference_distance_m)
    - shadow_noise_db (Gaussian shadowing term X_sigma)
    - path_loss_db_with_noise
    - signal_strength_dbm (RSSI)
    """
    if reference_distance_m <= 0:
        raise ValueError("reference_distance_m must be > 0.")

    links_df = extract_rssi_channel_inputs(data_dir)
    scenario_params_df = create_scenario_rssi_parameters(links_df, seed=seed)

    rssi_df = links_df.merge(
        scenario_params_df,
        on=["scenario_id", "env_type"],
        how="left",
        validate="many_to_one",
    )

    if (rssi_df["distance_m"] <= 0).any():
        bad_count = int((rssi_df["distance_m"] <= 0).sum())
        raise ValueError(f"distance_m must be > 0 for RSSI calculations. Found {bad_count} invalid rows.")

    np_rng = np.random.default_rng(seed)

    if reference_a_dbm is not None:
        rssi_df["initial_signal_strength_dbm"] = float(reference_a_dbm)

    # Shadowing is modelled as zero-mean Gaussian noise in the log-domain with scenario-specific sigma.
    rssi_df["shadow_noise_db"] = np_rng.normal(
        loc=0.0,
        scale=rssi_df["shadow_sigma_db"].astype(float).to_numpy(),
        size=len(rssi_df),
    )

    path_loss_increment_db = (
        10.0
        * rssi_df["path_loss_exponent_n"].astype(float)
        * np.log10(rssi_df["distance_m"].astype(float) / reference_distance_m)
    )

    rssi_df["path_loss_db_with_noise"] = (
        path_loss_increment_db
        + rssi_df["shadow_noise_db"].astype(float)
    )

    rssi_df["signal_strength_dbm"] = (
        rssi_df["initial_signal_strength_dbm"].astype(float)
        - rssi_df["path_loss_db_with_noise"].astype(float)
    )

    ordered_cols = [
        *[c for c in links_df.columns],
        "path_loss_exponent_n",
        "shadow_sigma_db",
        "initial_signal_strength_dbm",
        "shadow_noise_db",
        "path_loss_db_with_noise",
        "signal_strength_dbm",
    ]
    return rssi_df[ordered_cols].copy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build RSSI/path-loss columns from generated network scenario parquet files."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="generated_network_scenarios",
        help="Directory containing summary.parquet and channels.parquet.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed for reproducible n/sigma/noise sampling.",
    )
    parser.add_argument(
        "--reference-a-dbm",
        type=float,
        default=None,
        help="Optional fixed A reference RSSI in dBm at d0. Default: sampled per scenario.",
    )
    parser.add_argument(
        "--reference-distance-m",
        type=float,
        default=1.0,
        help="Reference distance d0 in meters (default: 1.0).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output parquet path. Default: <data-dir>/channels_rssi.parquet",
    )
    parser.add_argument(
        "--overwrite-channels",
        action="store_true",
        help="If set, write output directly to <data-dir>/channels.parquet.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if args.overwrite_channels:
        output_path = data_dir / "channels.parquet"
    elif args.output:
        output_path = Path(args.output)
    else:
        output_path = data_dir / "channels_rssi.parquet"

    rssi_df = build_rssi_base_table(
        data_dir=data_dir,
        seed=args.seed,
        reference_a_dbm=args.reference_a_dbm,
        reference_distance_m=args.reference_distance_m,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rssi_df.to_parquet(output_path, index=False)

    print(f"Wrote {len(rssi_df)} RSSI rows to {output_path}")
    if args.seed is None:
        print("Seed: none (sampling varies across runs).")
    else:
        print(f"Seed: {args.seed} (reproducible sampling).")


if __name__ == "__main__":
    main()
