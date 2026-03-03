from __future__ import annotations
import argparse
import math
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

    RSSI_{i,j} = P_t + G_{Tx} + G_{Rx} - PL_{i,j}

        where:
        - P_t = is transmit power (dBm)
        - G_{Tx} / G_{Rx} are antenna gains (dBi)
        - PL_{i,j} = total path loss (dB)
    
    - Path Loss exponent: The average rate at which signal power decays with distance and depends on the propagation environment.

    - Log-Distance Path Loss Model: The large-scale average path loss between two devices separated by distance (d)
      is modelled using the log-distance path loss model.

    - Log-Normal Shadowing (Gaussian Noise): To model environmental variability and measurement uncertainty, log-normal shadowing
      is applied by adding a zero-mean Gaussian random variable in the logarithmic domain.


RUNNIG SCRIPT
-------------
python3 "Data Generation/RSSI/RSSI_envs.py" --data-dir "generated_network_scenarios"
python3 "Data Generation/RSSI/RSSI_envs.py" --data-dir "generated_network_scenarios" --seed 42 --tx-power-dbm 0.0

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

"""
We will run through every channel in an environment and perform the same calculations. 

So we will have General, RSSI, TDOA and DOA channel columns. 

The columns will only hold the extra parameters which we cannot find in the general column. 

Later on in the ML pipeline we can define how we output all the values to train the model.
"""

# Returns the DB signal strength path loss
# Log-Distance Path Loss Model formula: PL(d) = PL(d_0) + 10 n \log_{10}\!\left(\frac{d}{d_0}\right)
# PL(d_0) = PL(d_0) = 20 \log_{10}\!\left(\frac{4\pi d_0}{\lambda}\right)
# lambda = \lambda = \frac{c}{f}
# c = speed of light
# f = frequency is 5GHZ
def _calc_log_distance_path_loss_model(path_loss_exponent: float, distance: float) -> float:

    #1) Free-Space Path Loss at the reference distance
    frequency: int = 5e9 #5,000,000,000 hertz
    speed_of_light: int = 2.99792458e8 # 299 792 458 m/s
    reference_distance_d0: int = 1 # meter
    wavelength: int = speed_of_light / frequency
    pl_d_0: float = 20 * np.log10(4 * np.pi * reference_distance_d0 / wavelength)
    
    
    #2) Log-Distance Path Loss Model
    pl_d = pl_d_0 + 10 * path_loss_exponent * np.log10(distance / reference_distance_d0)

    return pl_d


def _calc_log_normal_shadowing_gaussian_noise(path_loss_at_distance_d: float, noise_std_dev: float) -> float:

    x_sigma = np.random.normal(loc=0.0, scale=noise_std_dev)

    pl_d = path_loss_at_distance_d + x_sigma

    return pl_d


#def _set_transmit_power(): 

#def _set_antenna_gains(): 




def extract_rssi_channel_inputs(data_dir: Union[str, Path]) -> pd.DataFrame
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

    rssi_network_scenario["path_loss_exponent_n"] = rssi_network_scenario["env_type"].map(sample_n)
    rssi_network_scenario["shadow_sigma_db"] = rssi_network_scenario["env_type"].map(sample_sigma)
    return rssi_network_scenario


def _calculate_path(
) -> pd.Series:
    """Compute RSSI reference A(d0) from frequency using free-space loss at d0."""
    speed_light_m_per_s = 299_792_458.0
    freq_hz = freq_mhz.astype(float) * 1e6
    pl_d0_db = 20.0 * np.log10((4.0 * math.pi * reference_distance_m * freq_hz) / speed_light_m_per_s)
    return float(tx_power_dbm) - pl_d0_db


def build_rssi_base_table(
    data_dir: Union[str, Path],
    seed: int | None = None,
    tx_power_dbm: float = 0.0,
    reference_distance_m: float = 1.0,
) -> pd.DataFrame:
    """
    Return per-channel table with original channel fields plus RSSI/path-loss attributes.

    Added columns:
    - path_loss_exponent_n (scenario-level)
    - shadow_sigma_db (scenario-level)
    - shadow_noise_db (Gaussian shadowing term X_sigma)
    - path_loss_db_with_noise
    - initial_signal_strength_dbm
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

    freq_col = "initial_freq_mhz" if "initial_freq_mhz" in rssi_df.columns else "freq_mhz"
    rssi_df["initial_signal_strength_dbm"] = _reference_rssi_dbm_from_freq(
        freq_mhz=rssi_df[freq_col],
        tx_power_dbm=tx_power_dbm,
        reference_distance_m=reference_distance_m,
    )

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
        "--tx-power-dbm",
        type=float,
        default=0.0,
        help="Transmit power in dBm used to compute A at d0 from channel frequency.",
    )
    parser.add_argument(
        "--reference-distance-m",
        type=float,
        default=1.0,
        help="Reference distance d0 in meters used for A and path-loss terms.",
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
        tx_power_dbm=args.tx_power_dbm,
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
