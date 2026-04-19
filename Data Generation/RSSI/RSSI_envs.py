"""
RSSI ENVIRONMENT LAYER
----------------------

This module derives RSSI-specific columns from the shared per-link geometry
table produced by `link_factory.py`.

Input tables:
- env_summary.parquet
- links.parquet

Output table:
- links_rssi.parquet by default

The resulting rows remain one row per antenna-target link, which is the right
shape for later pivoting into ML feature matrices.

Usage examples:
python3 "Data Generation/RSSI/RSSI_envs.py" --data-dir "generated_network_scenarios"
python3 "Data Generation/RSSI/RSSI_envs.py" --data-dir "generated_network_scenarios" --seed 42
python3 "Data Generation/RSSI/RSSI_envs.py" \
  --data-dir "Data Generation/generated_network_scenarios" \
  --tx-power-dbm 20.0 \
  --tx-gain-dbi 3.0
  
@author: Giuliana Emberson
@date: 7th of May 2026

"""

from __future__ import annotations
import argparse
import math
import random
from pathlib import Path
from typing import Union
import numpy as np
import pandas as pd


PATH_LOSS_EXPONENT_RANGES = {
    "outdoor": (2.7, 3.5),
    "indoor_los": (1.6, 1.8),
    "indoor_nlos": (4.0, 6.0),
}

# Aligned to the project RSSI README's typical shadow-fading values.
SHADOW_SIGMA_DB_RANGES = {
    "outdoor": (1.0, 2.0),
    "indoor_los": (2.0, 4.0),
    "indoor_nlos": (4.0, 8.0),
}


def _normalize_env_type(env: str) -> str:
    normalized = str(env).strip().lower()
    if normalized not in PATH_LOSS_EXPONENT_RANGES:
        raise ValueError(f"Unsupported RSSI environment type: {env}")
    return normalized


def _resolve_rssi_env_type(env_type: str, link_state: str) -> str:
    env = str(env_type).strip().lower()
    state = str(link_state).strip().upper()

    if env in PATH_LOSS_EXPONENT_RANGES:
        return env

    if env == "outdoor":
        return "outdoor"

    if env == "indoor":
        return "indoor_los" if state == "LOS" else "indoor_nlos"

    raise ValueError(f"Unsupported generator env_type: {env_type}")


def path_loss_exponent_given_env_type(env: str, *, rng: random.Random | None = None) -> float:
    normalized = _normalize_env_type(env)
    lo, hi = PATH_LOSS_EXPONENT_RANGES[normalized]
    sampler = rng or random
    return sampler.uniform(lo, hi)


def shadow_sigma_given_env_type(env: str, *, rng: random.Random | None = None) -> float:
    normalized = _normalize_env_type(env)
    lo, hi = SHADOW_SIGMA_DB_RANGES[normalized]
    sampler = rng or random
    return sampler.uniform(lo, hi)


def _reference_rssi_dbm_from_freq(
    *,
    freq_mhz: Union[float, pd.Series],
    tx_power_dbm: float,
    tx_gain_dbi: float,
    rx_gain_dbi: float,
    reference_distance_m: float,
) -> Union[float, pd.Series]:
    speed_light_m_per_s = 299_792_458.0
    freq_hz = pd.Series(freq_mhz, copy=False).astype(float) * 1e6
    pl_d0_db = 20.0 * np.log10((4.0 * math.pi * reference_distance_m * freq_hz) / speed_light_m_per_s)
    return float(tx_power_dbm + tx_gain_dbi + rx_gain_dbi) - pl_d0_db


def extract_rssi_link_inputs(data_dir: Union[str, Path]) -> pd.DataFrame:
    """
    Return one row per antenna-target link with the scenario environment joined in.

    Required columns:
    - scenario_id
    - distance_m
    - link_state
    """
    data_path = Path(data_dir)
    summary_path = data_path / "env_summary.parquet"
    links_path = data_path / "links.parquet"

    summary_env_df = pd.read_parquet(summary_path, columns=["scenario_id", "env_type"])
    links_df = pd.read_parquet(links_path)

    required_link_cols = {"scenario_id", "distance_m", "link_state"}
    missing_cols = required_link_cols.difference(links_df.columns)
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

    merged_df["rssi_env_type"] = [
        _resolve_rssi_env_type(env_type, link_state)
        for env_type, link_state in zip(merged_df["env_type"], merged_df["link_state"])
    ]
    return merged_df.copy()


def create_scenario_rssi_parameters(
    link_inputs_df: pd.DataFrame,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Sample RSSI parameters once per scenario and per derived RSSI environment
    class. Indoor scenarios can therefore have distinct LOS and NLOS parameter
    draws while preserving scenario-level consistency.
    """
    rng = random.Random(seed)
    scenario_rssi_df = link_inputs_df[["scenario_id", "rssi_env_type"]].drop_duplicates().copy()

    scenario_rssi_df["path_loss_exponent_n"] = scenario_rssi_df["rssi_env_type"].map(
        lambda env: path_loss_exponent_given_env_type(env, rng=rng)
    )
    scenario_rssi_df["shadow_sigma_db"] = scenario_rssi_df["rssi_env_type"].map(
        lambda env: shadow_sigma_given_env_type(env, rng=rng)
    )
    return scenario_rssi_df


def build_rssi_base_table(
    data_dir: Union[str, Path],
    *,
    seed: int | None = None,
    freq_mhz: float = 5500.0,
    tx_power_dbm: float = 0.0,
    tx_gain_dbi: float = 0.0,
    rx_gain_dbi: float = 0.0,
    reference_distance_m: float = 1.0,
    wall_loss_db: float = 0.0,
    human_loss_db: float = 0.0,
) -> pd.DataFrame:
    """
    Return the shared link table with RSSI/path-loss columns added.

    Added columns:
    - rssi_env_type
    - path_loss_exponent_n
    - shadow_sigma_db
    - freq_mhz
    - tx_power_dbm
    - tx_gain_dbi
    - rx_gain_dbi
    - reference_distance_m
    - initial_signal_strength_dbm
    - wall_attenuation_db
    - human_attenuation_db
    - obstacle_attenuation_db
    - shadow_noise_db
    - path_loss_db_with_noise
    - signal_strength_dbm
    """
    if reference_distance_m <= 0:
        raise ValueError("reference_distance_m must be > 0.")
    if freq_mhz <= 0:
        raise ValueError("freq_mhz must be > 0.")
    if wall_loss_db < 0 or human_loss_db < 0:
        raise ValueError("wall_loss_db and human_loss_db must be >= 0.")

    links_df = extract_rssi_link_inputs(data_dir)
    scenario_params_df = create_scenario_rssi_parameters(links_df, seed=seed)

    rssi_df = links_df.merge(
        scenario_params_df,
        on=["scenario_id", "rssi_env_type"],
        how="left",
        validate="many_to_one",
    )

    if (rssi_df["distance_m"] <= 0).any():
        bad_count = int((rssi_df["distance_m"] <= 0).sum())
        raise ValueError(f"distance_m must be > 0 for RSSI calculations. Found {bad_count} invalid rows.")

    np_rng = np.random.default_rng(seed)

    rssi_df["freq_mhz"] = float(freq_mhz)
    rssi_df["tx_power_dbm"] = float(tx_power_dbm)
    rssi_df["tx_gain_dbi"] = float(tx_gain_dbi)
    rssi_df["rx_gain_dbi"] = float(rx_gain_dbi)
    rssi_df["reference_distance_m"] = float(reference_distance_m)

    rssi_df["initial_signal_strength_dbm"] = _reference_rssi_dbm_from_freq(
        freq_mhz=rssi_df["freq_mhz"],
        tx_power_dbm=tx_power_dbm,
        tx_gain_dbi=tx_gain_dbi,
        rx_gain_dbi=rx_gain_dbi,
        reference_distance_m=reference_distance_m,
    )

    wall_counts = (
        rssi_df["wall_blocker_count"].astype(float)
        if "wall_blocker_count" in rssi_df.columns
        else pd.Series(0.0, index=rssi_df.index)
    )
    human_counts = (
        rssi_df["human_blocker_count"].astype(float)
        if "human_blocker_count" in rssi_df.columns
        else pd.Series(0.0, index=rssi_df.index)
    )

    rssi_df["wall_attenuation_db"] = wall_counts * float(wall_loss_db)
    rssi_df["human_attenuation_db"] = human_counts * float(human_loss_db)
    rssi_df["obstacle_attenuation_db"] = (
        rssi_df["wall_attenuation_db"] + rssi_df["human_attenuation_db"]
    )

    # Shadowing is modelled as zero-mean Gaussian noise in the log-domain with
    # sigma sampled per scenario and per derived RSSI environment class.
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
        + rssi_df["obstacle_attenuation_db"].astype(float)
        + rssi_df["shadow_noise_db"].astype(float)
    )

    rssi_df["signal_strength_dbm"] = (
        rssi_df["initial_signal_strength_dbm"].astype(float)
        - rssi_df["path_loss_db_with_noise"].astype(float)
    )

    added_cols = [
        "path_loss_exponent_n",
        "shadow_sigma_db",
        "freq_mhz",
        "tx_power_dbm",
        "tx_gain_dbi",
        "rx_gain_dbi",
        "reference_distance_m",
        "initial_signal_strength_dbm",
        "wall_attenuation_db",
        "human_attenuation_db",
        "obstacle_attenuation_db",
        "shadow_noise_db",
        "path_loss_db_with_noise",
        "signal_strength_dbm",
    ]
    ordered_cols = [*links_df.columns, *[column for column in added_cols if column not in links_df.columns]]
    return rssi_df[ordered_cols].copy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build RSSI/path-loss columns from env_summary.parquet and links.parquet."
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
        help="Optional seed for reproducible path-loss, sigma, and shadow-noise sampling.",
    )
    parser.add_argument(
        "--freq-mhz",
        type=float,
        default=5500.0,
        help="Carrier frequency in MHz used for the RSSI reference term.",
    )
    parser.add_argument(
        "--tx-power-dbm",
        type=float,
        default=0.0,
        help="Transmit power in dBm used to compute RSSI at the reference distance.",
    )
    parser.add_argument(
        "--tx-gain-dbi",
        type=float,
        default=0.0,
        help="Transmit antenna gain in dBi.",
    )
    parser.add_argument(
        "--rx-gain-dbi",
        type=float,
        default=0.0,
        help="Receive antenna gain in dBi.",
    )
    parser.add_argument(
        "--reference-distance-m",
        type=float,
        default=1.0,
        help="Reference distance d0 in meters used for the RSSI reference term.",
    )
    parser.add_argument(
        "--wall-loss-db",
        type=float,
        default=0.0,
        help="Optional extra attenuation applied per intersected wall.",
    )
    parser.add_argument(
        "--human-loss-db",
        type=float,
        default=0.0,
        help="Optional extra attenuation applied per intersected human obstacle.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output parquet path. Default: <data-dir>/links_rssi.parquet",
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
        output_path = data_dir / "links_rssi.parquet"

    rssi_df = build_rssi_base_table(
        data_dir=data_dir,
        seed=args.seed,
        freq_mhz=args.freq_mhz,
        tx_power_dbm=args.tx_power_dbm,
        tx_gain_dbi=args.tx_gain_dbi,
        rx_gain_dbi=args.rx_gain_dbi,
        reference_distance_m=args.reference_distance_m,
        wall_loss_db=args.wall_loss_db,
        human_loss_db=args.human_loss_db,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rssi_df.to_parquet(output_path, index=False)

    print(f"Wrote {len(rssi_df)} RSSI link rows to {output_path}")
    if args.seed is None:
        print("Seed: none (sampling varies across runs).")
    else:
        print(f"Seed: {args.seed} (reproducible sampling).")


if __name__ == "__main__":
    main()
