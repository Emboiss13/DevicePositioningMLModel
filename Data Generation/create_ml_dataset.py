"""
LABELLED ML DATASET BUILDER
---------------------------

Construct one supervised-learning row per scenario-target pair by merging:
- scenario/environment features
- antenna-layout summaries
- shared link geometry summaries
- RSSI, TDOA, and DOA/AOA measurement summaries
- conventional RSSI/TDOA/DOA position estimates

Input tables:
- env_summary.parquet
- antennas.parquet
- links.parquet
- links_rssi.parquet
- links_tdoa.parquet
- links_doa.parquet
- position_estimates.parquet

Output table:
- ml_dataset.parquet by default

Usage examples:
python3 "Data Generation/create_ml_dataset.py" --data-dir "generated_network_scenarios"
python3 "Data Generation/create_ml_dataset.py" \
  --data-dir "Data Generation/generated_network_scenarios" \
  --output "Data Generation/generated_network_scenarios/ml_dataset.parquet"
python3 "Data Generation/create_ml_dataset.py" \
  --data-dir "Data Generation/generated_network_scenarios"

python3 "Data Generation/create_ml_dataset.py" --data-dir "Data Generation/generated_network_scenarios_with_plots"

@author: Giuliana Emberson
@date: 7th of May 2026

"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Iterable, Union
import numpy as np
import pandas as pd


KEY_COLUMNS = ["scenario_id", "target_id"]


def _require_columns(df: pd.DataFrame, required: Iterable[str], table_name: str) -> None:
    missing = set(required).difference(df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"{table_name} is missing required columns: {missing_list}")


def _read_table(data_dir: Union[str, Path], table_name: str, required: Iterable[str]) -> pd.DataFrame:
    path = Path(data_dir) / f"{table_name}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run the corresponding data-generation stage first."
        )
    df = pd.read_parquet(path)
    _require_columns(df, required, f"{table_name}.parquet")
    return df


def _std_or_zero(series: pd.Series) -> float:
    value = series.std()
    if pd.isna(value):
        return 0.0
    return float(value)


def _scenario_features(env_summary_df: pd.DataFrame) -> pd.DataFrame:
    preferred_cols = [
        "scenario_id",
        "seed",
        "area",
        "env_type",
        "width",
        "height",
        "x_domain_min",
        "x_domain_max",
        "y_range_min",
        "y_range_max",
        "antenna_count",
        "human_count",
        "floor_plan_room_count",
        "floor_plan_patio_count",
        "floor_plan_element_count",
        "target_count",
        "grid_cell_count",
        "valid_grid_cell_count",
        "target_grid_rows",
        "target_grid_cols",
        "target_requested_max_cell_size",
        "target_cell_width",
        "target_cell_height",
        "link_count",
    ]
    available_cols = [column for column in preferred_cols if column in env_summary_df.columns]
    scenario_df = env_summary_df[available_cols].drop_duplicates().copy()
    duplicates = scenario_df.duplicated("scenario_id", keep=False)
    if duplicates.any():
        duplicate_count = int(duplicates.sum())
        raise ValueError(f"env_summary.parquet has {duplicate_count} duplicate scenario rows.")
    return scenario_df


def _antenna_features(antennas_df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        antennas_df,
        ["scenario_id", "antenna_id", "x", "y", "coverage_radius"],
        "antennas.parquet",
    )
    grouped = antennas_df.groupby("scenario_id", sort=True)
    features_df = grouped.agg(
        antenna_layout_count=("antenna_id", "nunique"),
        antenna_x_mean=("x", "mean"),
        antenna_y_mean=("y", "mean"),
        antenna_x_min=("x", "min"),
        antenna_x_max=("x", "max"),
        antenna_y_min=("y", "min"),
        antenna_y_max=("y", "max"),
        antenna_coverage_radius_mean_m=("coverage_radius", "mean"),
        antenna_coverage_radius_min_m=("coverage_radius", "min"),
        antenna_coverage_radius_max_m=("coverage_radius", "max"),
    ).reset_index()
    spread_df = grouped.agg(
        antenna_x_std=("x", _std_or_zero),
        antenna_y_std=("y", _std_or_zero),
    ).reset_index()
    return features_df.merge(spread_df, on="scenario_id", how="left", validate="one_to_one")


def _link_features(links_df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        links_df,
        [
            "scenario_id",
            "target_id",
            "antenna_id",
            "distance_m",
            "link_state",
            "wall_blocker_count",
            "human_blocker_count",
            "total_blocker_count",
        ],
        "links.parquet",
    )
    grouped = links_df.groupby(KEY_COLUMNS, sort=True)
    features_df = grouped.agg(
        link_anchor_count=("antenna_id", "nunique"),
        link_distance_mean_m=("distance_m", "mean"),
        link_distance_min_m=("distance_m", "min"),
        link_distance_max_m=("distance_m", "max"),
        link_wall_blocker_mean=("wall_blocker_count", "mean"),
        link_wall_blocker_max=("wall_blocker_count", "max"),
        link_human_blocker_mean=("human_blocker_count", "mean"),
        link_human_blocker_max=("human_blocker_count", "max"),
        link_total_blocker_mean=("total_blocker_count", "mean"),
        link_total_blocker_max=("total_blocker_count", "max"),
    ).reset_index()
    spread_df = grouped.agg(
        link_distance_std_m=("distance_m", _std_or_zero),
    ).reset_index()
    state_counts = (
        links_df.assign(
            link_los_count=(links_df["link_state"].astype(str).str.upper() == "LOS").astype(int),
            link_nlos_count=(links_df["link_state"].astype(str).str.upper() == "NLOS").astype(int),
        )
        .groupby(KEY_COLUMNS, sort=True)[["link_los_count", "link_nlos_count"]]
        .sum()
        .reset_index()
    )
    return (
        features_df.merge(spread_df, on=KEY_COLUMNS, how="left", validate="one_to_one")
        .merge(state_counts, on=KEY_COLUMNS, how="left", validate="one_to_one")
    )


def _rssi_features(rssi_df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        rssi_df,
        [
            "scenario_id",
            "target_id",
            "signal_strength_dbm",
            "path_loss_db_with_noise",
            "path_loss_exponent_n",
            "shadow_sigma_db",
        ],
        "links_rssi.parquet",
    )
    optional_cols = {
        "obstacle_attenuation_db": 0.0,
        "wall_attenuation_db": 0.0,
        "human_attenuation_db": 0.0,
    }
    prepared_df = rssi_df.copy()
    for column, default in optional_cols.items():
        if column not in prepared_df.columns:
            prepared_df[column] = default

    grouped = prepared_df.groupby(KEY_COLUMNS, sort=True)
    features_df = grouped.agg(
        rssi_measurement_count=("signal_strength_dbm", "count"),
        rssi_signal_mean_dbm=("signal_strength_dbm", "mean"),
        rssi_signal_min_dbm=("signal_strength_dbm", "min"),
        rssi_signal_max_dbm=("signal_strength_dbm", "max"),
        rssi_path_loss_mean_db=("path_loss_db_with_noise", "mean"),
        rssi_path_loss_min_db=("path_loss_db_with_noise", "min"),
        rssi_path_loss_max_db=("path_loss_db_with_noise", "max"),
        rssi_path_loss_exponent_mean=("path_loss_exponent_n", "mean"),
        rssi_shadow_sigma_mean_db=("shadow_sigma_db", "mean"),
        rssi_obstacle_attenuation_mean_db=("obstacle_attenuation_db", "mean"),
        rssi_obstacle_attenuation_max_db=("obstacle_attenuation_db", "max"),
        rssi_wall_attenuation_mean_db=("wall_attenuation_db", "mean"),
        rssi_human_attenuation_mean_db=("human_attenuation_db", "mean"),
    ).reset_index()
    spread_df = grouped.agg(
        rssi_signal_std_dbm=("signal_strength_dbm", _std_or_zero),
        rssi_path_loss_std_db=("path_loss_db_with_noise", _std_or_zero),
    ).reset_index()
    return features_df.merge(spread_df, on=KEY_COLUMNS, how="left", validate="one_to_one")


def _tdoa_features(tdoa_df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        tdoa_df,
        [
            "scenario_id",
            "target_id",
            "observed_tdoa_ns",
            "delta_distance_m",
            "reference_distance_m",
            "comparison_distance_m",
            "tdoa_noise_sigma_ns",
        ],
        "links_tdoa.parquet",
    )
    grouped = tdoa_df.groupby(KEY_COLUMNS, sort=True)
    features_df = grouped.agg(
        tdoa_measurement_count=("observed_tdoa_ns", "count"),
        tdoa_observed_mean_ns=("observed_tdoa_ns", "mean"),
        tdoa_observed_min_ns=("observed_tdoa_ns", "min"),
        tdoa_observed_max_ns=("observed_tdoa_ns", "max"),
        tdoa_delta_distance_mean_m=("delta_distance_m", "mean"),
        tdoa_delta_distance_min_m=("delta_distance_m", "min"),
        tdoa_delta_distance_max_m=("delta_distance_m", "max"),
        tdoa_reference_distance_mean_m=("reference_distance_m", "mean"),
        tdoa_comparison_distance_mean_m=("comparison_distance_m", "mean"),
        tdoa_noise_sigma_mean_ns=("tdoa_noise_sigma_ns", "mean"),
    ).reset_index()
    spread_df = grouped.agg(
        tdoa_observed_std_ns=("observed_tdoa_ns", _std_or_zero),
        tdoa_delta_distance_std_m=("delta_distance_m", _std_or_zero),
    ).reset_index()
    return features_df.merge(spread_df, on=KEY_COLUMNS, how="left", validate="one_to_one")


def _doa_features(doa_df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        doa_df,
        [
            "scenario_id",
            "target_id",
            "observed_bearing_rad",
            "observed_doa_rad",
            "doa_noise_sigma_deg",
        ],
        "links_doa.parquet",
    )
    prepared_df = doa_df.copy()
    prepared_df["doa_observed_bearing_sin"] = np.sin(
        prepared_df["observed_bearing_rad"].astype(float)
    )
    prepared_df["doa_observed_bearing_cos"] = np.cos(
        prepared_df["observed_bearing_rad"].astype(float)
    )
    prepared_df["doa_observed_doa_sin"] = np.sin(
        prepared_df["observed_doa_rad"].astype(float)
    )
    prepared_df["doa_observed_doa_cos"] = np.cos(
        prepared_df["observed_doa_rad"].astype(float)
    )

    grouped = prepared_df.groupby(KEY_COLUMNS, sort=True)
    return grouped.agg(
        doa_measurement_count=("observed_bearing_rad", "count"),
        doa_observed_bearing_sin_mean=("doa_observed_bearing_sin", "mean"),
        doa_observed_bearing_cos_mean=("doa_observed_bearing_cos", "mean"),
        doa_observed_doa_sin_mean=("doa_observed_doa_sin", "mean"),
        doa_observed_doa_cos_mean=("doa_observed_doa_cos", "mean"),
        doa_noise_sigma_mean_deg=("doa_noise_sigma_deg", "mean"),
    ).reset_index()


def _validate_one_row_per_target(dataset_df: pd.DataFrame) -> None:
    duplicate_rows = dataset_df.duplicated(KEY_COLUMNS, keep=False)
    if duplicate_rows.any():
        duplicate_count = int(duplicate_rows.sum())
        raise ValueError(f"ML dataset has {duplicate_count} duplicate scenario-target rows.")
    if dataset_df[["target_x", "target_y"]].isna().any().any():
        raise ValueError("ML dataset contains missing target_x or target_y labels.")


def build_ml_dataset(data_dir: Union[str, Path]) -> pd.DataFrame:
    env_summary_df = _read_table(data_dir, "env_summary", ["scenario_id", "env_type"])
    antennas_df = _read_table(data_dir, "antennas", ["scenario_id", "antenna_id"])
    links_df = _read_table(data_dir, "links", ["scenario_id", "target_id"])
    rssi_df = _read_table(data_dir, "links_rssi", ["scenario_id", "target_id"])
    tdoa_df = _read_table(data_dir, "links_tdoa", ["scenario_id", "target_id"])
    doa_df = _read_table(data_dir, "links_doa", ["scenario_id", "target_id"])
    estimates_df = _read_table(
        data_dir,
        "position_estimates",
        ["scenario_id", "target_id", "target_x", "target_y"],
    )

    duplicate_estimates = estimates_df.duplicated(KEY_COLUMNS, keep=False)
    if duplicate_estimates.any():
        duplicate_count = int(duplicate_estimates.sum())
        raise ValueError(
            f"position_estimates.parquet has {duplicate_count} duplicate scenario-target rows."
        )

    dataset_df = estimates_df.copy()
    dataset_df = dataset_df.merge(
        _scenario_features(env_summary_df),
        on="scenario_id",
        how="left",
        validate="many_to_one",
    )
    dataset_df = dataset_df.merge(
        _antenna_features(antennas_df),
        on="scenario_id",
        how="left",
        validate="many_to_one",
    )
    dataset_df = dataset_df.merge(
        _link_features(links_df),
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    dataset_df = dataset_df.merge(
        _rssi_features(rssi_df),
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    dataset_df = dataset_df.merge(
        _tdoa_features(tdoa_df),
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    dataset_df = dataset_df.merge(
        _doa_features(doa_df),
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )

    _validate_one_row_per_target(dataset_df)
    return dataset_df.copy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build one labelled ML dataset row per scenario-target pair."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="generated_network_scenarios",
        help="Directory containing generated parquet tables.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output parquet path. Default: <data-dir>/ml_dataset.parquet",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_path = Path(args.output) if args.output else data_dir / "ml_dataset.parquet"
    ml_dataset_df = build_ml_dataset(data_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ml_dataset_df.to_parquet(output_path, index=False)

    print(f"Wrote {len(ml_dataset_df)} labelled ML rows to {output_path}")
    print("Labels: target_x, target_y")


if __name__ == "__main__":
    main()
