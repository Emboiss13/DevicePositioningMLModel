"""
DATA GENERATION OUTPUT VALIDATOR
--------------------------------

Validate the generated parquet tables before starting the ML pipeline.

Checks:
- Shared link row counts match target_count * antenna_count.
- RSSI and DOA measurement row counts match links.parquet.
- TDOA row counts match target_count * (antenna_count - 1).
- Required measurement columns exist and are populated.
- position_estimates.parquet and ml_dataset.parquet contain one row per target.
- ML labels and conventional estimate/error columns are present.

Usage:
python3 "Data Generation/validate_generation_outputs.py" --data-dir "generated_network_scenarios"
python3 "Data Generation/validate_generation_outputs.py" \
  --data-dir "Data Generation/generated_network_scenarios"

python3 "Data Generation/validate_generation_outputs.py" --data-dir "Data Generation/generated_network_scenarios_with_plots"

@author: Giuliana Emberson
@date: 7th of May 2026

"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Iterable, Union
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
        raise FileNotFoundError(f"Missing required table: {path}")
    df = pd.read_parquet(path)
    _require_columns(df, required, f"{table_name}.parquet")
    return df


def _check_no_missing(df: pd.DataFrame, columns: Iterable[str], table_name: str) -> None:
    columns = list(columns)
    missing = df[columns].isna().sum()
    bad = missing[missing > 0]
    if not bad.empty:
        details = ", ".join(f"{column}={count}" for column, count in bad.items())
        raise ValueError(f"{table_name} has missing values in required columns: {details}")


def _check_unique_targets(df: pd.DataFrame, table_name: str) -> None:
    duplicates = df.duplicated(KEY_COLUMNS, keep=False)
    if duplicates.any():
        duplicate_count = int(duplicates.sum())
        raise ValueError(f"{table_name} has {duplicate_count} duplicate scenario-target rows.")


def _expected_counts(env_summary_df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        env_summary_df,
        ["scenario_id", "target_count", "antenna_count"],
        "env_summary.parquet",
    )
    counts_df = env_summary_df[["scenario_id", "target_count", "antenna_count"]].copy()
    counts_df["expected_link_count"] = (
        counts_df["target_count"].astype(int) * counts_df["antenna_count"].astype(int)
    )
    counts_df["expected_tdoa_count"] = (
        counts_df["target_count"].astype(int)
        * (counts_df["antenna_count"].astype(int) - 1)
    )
    return counts_df


def _check_count_by_scenario(
    actual_df: pd.DataFrame,
    counts_df: pd.DataFrame,
    *,
    expected_column: str,
    table_name: str,
    allow_fewer: bool = False,
) -> None:
    actual_counts = (
        actual_df.groupby("scenario_id", sort=True)
        .size()
        .rename("actual_count")
        .reset_index()
    )
    merged_df = counts_df[["scenario_id", expected_column]].merge(
        actual_counts,
        on="scenario_id",
        how="left",
        validate="one_to_one",
    )
    merged_df["actual_count"] = merged_df["actual_count"].fillna(0).astype(int)
    if allow_fewer:
        bad_df = merged_df[merged_df["actual_count"] > merged_df[expected_column]]
    else:
        bad_df = merged_df[merged_df["actual_count"] != merged_df[expected_column]]
    if not bad_df.empty:
        first_bad = bad_df.iloc[0]
        relation = "exceeded" if allow_fewer else "mismatch"
        raise ValueError(
            f"{table_name} count {relation} for {first_bad['scenario_id']}: "
            f"expected {'at most ' if allow_fewer else ''}{int(first_bad[expected_column])}, got {int(first_bad['actual_count'])}."
        )


def validate_generation_outputs(data_dir: Union[str, Path]) -> None:
    env_summary_df = _read_table(
        data_dir,
        "env_summary",
        ["scenario_id", "target_count", "antenna_count"],
    )
    links_df = _read_table(data_dir, "links", ["scenario_id", "target_id"])
    rssi_df = _read_table(
        data_dir,
        "links_rssi",
        ["scenario_id", "target_id", "signal_strength_dbm", "path_loss_db_with_noise"],
    )
    tdoa_df = _read_table(
        data_dir,
        "links_tdoa",
        ["scenario_id", "target_id", "observed_tdoa_ns"],
    )
    doa_df = _read_table(
        data_dir,
        "links_doa",
        ["scenario_id", "target_id", "observed_doa_deg", "observed_bearing_deg"],
    )
    estimates_df = _read_table(
        data_dir,
        "position_estimates",
        [
            "scenario_id",
            "target_id",
            "target_x",
            "target_y",
            "rssi_est_x",
            "rssi_est_y",
            "rssi_error_m",
            "tdoa_est_x",
            "tdoa_est_y",
            "tdoa_error_m",
            "doa_est_x",
            "doa_est_y",
            "doa_error_m",
        ],
    )
    ml_dataset_df = _read_table(
        data_dir,
        "ml_dataset",
        [
            "scenario_id",
            "target_id",
            "target_x",
            "target_y",
            "rssi_est_x",
            "rssi_est_y",
            "rssi_error_m",
            "tdoa_est_x",
            "tdoa_est_y",
            "tdoa_error_m",
            "doa_est_x",
            "doa_est_y",
            "doa_error_m",
        ],
    )

    counts_df = _expected_counts(env_summary_df)
    _check_count_by_scenario(
        links_df,
        counts_df,
        expected_column="expected_link_count",
        table_name="links.parquet",
    )
    _check_count_by_scenario(
        rssi_df,
        counts_df,
        expected_column="expected_link_count",
        table_name="links_rssi.parquet",
        allow_fewer=True,
    )
    _check_count_by_scenario(
        doa_df,
        counts_df,
        expected_column="expected_link_count",
        table_name="links_doa.parquet",
    )
    _check_count_by_scenario(
        tdoa_df,
        counts_df,
        expected_column="expected_tdoa_count",
        table_name="links_tdoa.parquet",
    )

    _check_no_missing(
        rssi_df,
        ["signal_strength_dbm", "path_loss_db_with_noise"],
        "links_rssi.parquet",
    )
    _check_no_missing(tdoa_df, ["observed_tdoa_ns"], "links_tdoa.parquet")
    if "is_doa_valid" in doa_df.columns:
        valid_doa_df = doa_df[doa_df["is_doa_valid"].fillna(False).astype(bool)]
        _check_no_missing(
            valid_doa_df,
            ["observed_doa_deg", "observed_bearing_deg"],
            "links_doa.parquet valid DOA rows",
        )
    else:
        _check_no_missing(
            doa_df,
            ["observed_doa_deg", "observed_bearing_deg"],
            "links_doa.parquet",
        )

    _check_unique_targets(estimates_df, "position_estimates.parquet")
    _check_unique_targets(ml_dataset_df, "ml_dataset.parquet")
    expected_target_total = int(counts_df["target_count"].sum())
    if len(estimates_df) != expected_target_total:
        raise ValueError(
            f"position_estimates.parquet row count mismatch: "
            f"expected {expected_target_total}, got {len(estimates_df)}."
        )
    if len(ml_dataset_df) != expected_target_total:
        raise ValueError(
            f"ml_dataset.parquet row count mismatch: "
            f"expected {expected_target_total}, got {len(ml_dataset_df)}."
        )

    _check_no_missing(estimates_df, ["target_x", "target_y"], "position_estimates.parquet")
    _check_no_missing(ml_dataset_df, ["target_x", "target_y"], "ml_dataset.parquet")

    for method in ("rssi", "tdoa", "doa"):
        success_col = f"{method}_success"
        if success_col in estimates_df.columns:
            successful_df = estimates_df[estimates_df[success_col].fillna(False)]
            if not successful_df.empty:
                _check_no_missing(
                    successful_df,
                    [f"{method}_est_x", f"{method}_est_y", f"{method}_error_m"],
                    "position_estimates.parquet",
                )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate generated data, measurement, estimate, and ML dataset tables."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="generated_network_scenarios",
        help="Directory containing generated parquet tables.",
    )
    args = parser.parse_args()

    validate_generation_outputs(args.data_dir)
    print(f"Validation passed for {args.data_dir}")


if __name__ == "__main__":
    main()
