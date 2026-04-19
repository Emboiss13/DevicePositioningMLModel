"""
CONVENTIONAL POSITION ESTIMATION
--------------------------------

Build target-position estimates from the RSSI, TDOA, and DOA/AOA measurement
tables produced by the data-generation pipeline.

Input tables:
- links.parquet
- links_rssi.parquet
- links_tdoa.parquet
- links_doa.parquet

Output table:
- position_estimates.parquet by default

The output contains one row per scenario-target pair. It keeps the ground-truth
target coordinates as labels and adds conventional method estimates plus error
metrics, ready to be merged into an ML feature table.

Usage examples:
python3 "Data Generation/position_estimation.py" --data-dir "generated_network_scenarios"
python3 "Data Generation/position_estimation.py" \
  --data-dir "Data Generation/generated_network_scenarios" \
  --output "Data Generation/generated_network_scenarios/position_estimates.parquet"

@author: Giuliana Emberson
@date: 7th of May 2026

"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd


KEY_COLUMNS = ["scenario_id", "target_id"]
DEFAULT_REFERENCE_DISTANCE_M = 1.0
EPSILON = 1e-9


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


def _target_truth_from_links(links_df: pd.DataFrame) -> pd.DataFrame:
    target_cols = [
        "scenario_id",
        "target_id",
        "target_label",
        "target_x",
        "target_y",
        "target_cell_id",
        "target_row_idx",
        "target_col_idx",
        "target_space_type",
        "target_room_id",
        "target_room_type",
        "target_patio_id",
    ]
    available_cols = [column for column in target_cols if column in links_df.columns]
    target_df = links_df[available_cols].drop_duplicates().copy()
    duplicate_keys = target_df.duplicated(KEY_COLUMNS, keep=False)
    if duplicate_keys.any():
        duplicate_count = int(duplicate_keys.sum())
        raise ValueError(
            f"links.parquet contains {duplicate_count} conflicting target-label rows."
        )
    return target_df


def _least_squares_range_position(
    anchors: np.ndarray,
    ranges_m: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> Tuple[Optional[np.ndarray], Optional[float], int]:
    valid = (
        np.isfinite(anchors).all(axis=1)
        & np.isfinite(ranges_m)
        & (ranges_m > 0)
    )
    if weights is not None:
        valid &= np.isfinite(weights) & (weights > 0)

    anchors = anchors[valid]
    ranges_m = ranges_m[valid]
    resolved_weights = weights[valid] if weights is not None else None
    anchor_count = len(anchors)
    if anchor_count < 3:
        return None, None, anchor_count

    ref_idx = int(np.argmin(ranges_m))
    ref_anchor = anchors[ref_idx]
    ref_range = ranges_m[ref_idx]
    other_mask = np.arange(anchor_count) != ref_idx
    other_anchors = anchors[other_mask]
    other_ranges = ranges_m[other_mask]

    a_matrix = 2.0 * (other_anchors - ref_anchor)
    b_vector = (
        (ref_range**2)
        - (other_ranges**2)
        + np.sum(other_anchors**2, axis=1)
        - np.sum(ref_anchor**2)
    )

    if resolved_weights is not None:
        other_weights = np.sqrt(resolved_weights[other_mask])
        a_matrix = a_matrix * other_weights[:, np.newaxis]
        b_vector = b_vector * other_weights

    try:
        solution, _, rank, _ = np.linalg.lstsq(a_matrix, b_vector, rcond=None)
    except np.linalg.LinAlgError:
        return None, None, anchor_count

    if rank < 2 or not np.isfinite(solution).all():
        return None, None, anchor_count

    position = _refine_range_position(solution, anchors, ranges_m, resolved_weights)
    residuals = np.linalg.norm(position - anchors, axis=1) - ranges_m
    rmse = float(np.sqrt(np.mean(residuals**2)))
    return position, rmse, anchor_count


def _refine_range_position(
    initial_position: np.ndarray,
    anchors: np.ndarray,
    ranges_m: np.ndarray,
    weights: Optional[np.ndarray],
    *,
    max_iterations: int = 25,
) -> np.ndarray:
    position = np.array(initial_position, dtype=float)
    sqrt_weights = np.sqrt(weights) if weights is not None else None

    def objective(candidate: np.ndarray) -> float:
        residuals = np.linalg.norm(candidate - anchors, axis=1) - ranges_m
        if sqrt_weights is not None:
            residuals = residuals * sqrt_weights
        return float(np.mean(residuals**2))

    best_score = objective(position)

    for _ in range(max_iterations):
        deltas = position - anchors
        distances = np.linalg.norm(deltas, axis=1)
        safe_distances = np.maximum(distances, EPSILON)
        residuals = distances - ranges_m
        jacobian = deltas / safe_distances[:, np.newaxis]

        if sqrt_weights is not None:
            weighted_jacobian = jacobian * sqrt_weights[:, np.newaxis]
            weighted_residuals = residuals * sqrt_weights
        else:
            weighted_jacobian = jacobian
            weighted_residuals = residuals

        try:
            step, _, rank, _ = np.linalg.lstsq(
                weighted_jacobian,
                -weighted_residuals,
                rcond=None,
            )
        except np.linalg.LinAlgError:
            break

        if rank < 2 or not np.isfinite(step).all():
            break

        accepted = False
        for damping in (1.0, 0.5, 0.25, 0.1, 0.05, 0.01):
            candidate = position + (step * damping)
            if not np.isfinite(candidate).all():
                continue
            candidate_score = objective(candidate)
            if candidate_score <= best_score:
                position = candidate
                best_score = candidate_score
                accepted = True
                break

        if not accepted:
            break

        if np.linalg.norm(step) < 1e-6:
            break

    return position


def _position_error_m(
    est_x: object,
    est_y: object,
    target_x: object,
    target_y: object,
) -> float:
    values = np.array([est_x, est_y, target_x, target_y], dtype=float)
    if not np.isfinite(values).all():
        return float("nan")
    return float(np.hypot(values[0] - values[2], values[1] - values[3]))


def _rssi_range_estimates_m(group: pd.DataFrame) -> np.ndarray:
    reference_distance = (
        group["reference_distance_m"].astype(float).to_numpy()
        if "reference_distance_m" in group.columns
        else np.full(len(group), DEFAULT_REFERENCE_DISTANCE_M, dtype=float)
    )
    path_loss_db = (
        group["initial_signal_strength_dbm"].astype(float).to_numpy()
        - group["signal_strength_dbm"].astype(float).to_numpy()
    )
    obstacle_loss_db = (
        group["obstacle_attenuation_db"].astype(float).to_numpy()
        if "obstacle_attenuation_db" in group.columns
        else np.zeros(len(group), dtype=float)
    )
    exponent_n = group["path_loss_exponent_n"].astype(float).to_numpy()
    compensated_loss_db = path_loss_db - obstacle_loss_db
    exponent = compensated_loss_db / np.maximum(10.0 * exponent_n, EPSILON)
    exponent = np.clip(exponent, -6.0, 6.0)
    return reference_distance * np.power(10.0, exponent)


def estimate_rssi_positions(rssi_df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        rssi_df,
        [
            "scenario_id",
            "target_id",
            "antenna_x",
            "antenna_y",
            "signal_strength_dbm",
            "initial_signal_strength_dbm",
            "path_loss_exponent_n",
        ],
        "links_rssi.parquet",
    )

    rows = []
    for (scenario_id, target_id), group in rssi_df.groupby(KEY_COLUMNS, sort=True):
        anchors = group[["antenna_x", "antenna_y"]].astype(float).to_numpy()
        ranges_m = _rssi_range_estimates_m(group)
        sigma = (
            group["shadow_sigma_db"].astype(float).to_numpy()
            if "shadow_sigma_db" in group.columns
            else np.ones(len(group), dtype=float)
        )
        weights = 1.0 / np.maximum(sigma, EPSILON) ** 2
        position, residual_rmse, anchor_count = _least_squares_range_position(
            anchors,
            ranges_m,
            weights=weights,
        )
        rows.append(
            {
                "scenario_id": scenario_id,
                "target_id": target_id,
                "rssi_anchor_count": anchor_count,
                "rssi_est_x": float(position[0]) if position is not None else np.nan,
                "rssi_est_y": float(position[1]) if position is not None else np.nan,
                "rssi_residual_rmse_m": residual_rmse,
                "rssi_success": position is not None,
            }
        )

    return pd.DataFrame(rows)


def _least_squares_tdoa_position(
    reference_anchor: np.ndarray,
    comparison_anchors: np.ndarray,
    delta_distances_m: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[float], int]:
    valid = (
        np.isfinite(comparison_anchors).all(axis=1)
        & np.isfinite(delta_distances_m)
        & np.isfinite(reference_anchor).all()
    )
    comparison_anchors = comparison_anchors[valid]
    delta_distances_m = delta_distances_m[valid]
    anchor_count = len(comparison_anchors) + 1
    if len(comparison_anchors) < 3:
        return None, None, anchor_count

    x0, y0 = reference_anchor
    xi = comparison_anchors[:, 0]
    yi = comparison_anchors[:, 1]
    delta = delta_distances_m

    a_matrix = np.column_stack(
        [
            2.0 * (x0 - xi),
            2.0 * (y0 - yi),
            -2.0 * delta,
        ]
    )
    b_vector = (delta**2) - (xi**2) + (x0**2) - (yi**2) + (y0**2)

    initial = np.mean(np.vstack([reference_anchor, comparison_anchors]), axis=0)
    try:
        solution, _, rank, _ = np.linalg.lstsq(a_matrix, b_vector, rcond=None)
        if rank >= 2 and np.isfinite(solution[:2]).all():
            initial = solution[:2]
    except np.linalg.LinAlgError:
        pass

    position = _refine_tdoa_position(
        initial,
        reference_anchor,
        comparison_anchors,
        delta_distances_m,
    )
    if position is None:
        return None, None, anchor_count

    reference_distance = np.linalg.norm(position - reference_anchor)
    comparison_distances = np.linalg.norm(position - comparison_anchors, axis=1)
    residuals = (comparison_distances - reference_distance) - delta_distances_m
    rmse = float(np.sqrt(np.mean(residuals**2)))
    return position, rmse, anchor_count


def _refine_tdoa_position(
    initial_position: np.ndarray,
    reference_anchor: np.ndarray,
    comparison_anchors: np.ndarray,
    delta_distances_m: np.ndarray,
    *,
    max_iterations: int = 35,
) -> Optional[np.ndarray]:
    position = np.array(initial_position, dtype=float)

    for _ in range(max_iterations):
        ref_delta = position - reference_anchor
        ref_distance = max(float(np.linalg.norm(ref_delta)), EPSILON)
        comp_delta = position - comparison_anchors
        comp_distances = np.maximum(np.linalg.norm(comp_delta, axis=1), EPSILON)

        residuals = (comp_distances - ref_distance) - delta_distances_m
        jacobian = (
            comp_delta / comp_distances[:, np.newaxis]
            - ref_delta / ref_distance
        )

        try:
            step, _, rank, _ = np.linalg.lstsq(jacobian, -residuals, rcond=None)
        except np.linalg.LinAlgError:
            return None

        if rank < 2 or not np.isfinite(step).all():
            return None

        position = position + step
        if np.linalg.norm(step) < 1e-6:
            break

    return position if np.isfinite(position).all() else None


def estimate_tdoa_positions(tdoa_df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        tdoa_df,
        [
            "scenario_id",
            "target_id",
            "reference_antenna_x",
            "reference_antenna_y",
            "comparison_antenna_x",
            "comparison_antenna_y",
            "observed_tdoa_ns",
            "propagation_speed_m_per_s",
        ],
        "links_tdoa.parquet",
    )

    rows = []
    for (scenario_id, target_id), group in tdoa_df.groupby(KEY_COLUMNS, sort=True):
        first = group.iloc[0]
        reference_anchor = np.array(
            [first["reference_antenna_x"], first["reference_antenna_y"]],
            dtype=float,
        )
        comparison_anchors = group[
            ["comparison_antenna_x", "comparison_antenna_y"]
        ].astype(float).to_numpy()
        propagation_speed = group["propagation_speed_m_per_s"].astype(float).to_numpy()
        delta_distances = (
            group["observed_tdoa_ns"].astype(float).to_numpy()
            * propagation_speed
            / 1e9
        )
        position, residual_rmse, anchor_count = _least_squares_tdoa_position(
            reference_anchor,
            comparison_anchors,
            delta_distances,
        )
        rows.append(
            {
                "scenario_id": scenario_id,
                "target_id": target_id,
                "tdoa_anchor_count": anchor_count,
                "tdoa_est_x": float(position[0]) if position is not None else np.nan,
                "tdoa_est_y": float(position[1]) if position is not None else np.nan,
                "tdoa_residual_rmse_m": residual_rmse,
                "tdoa_success": position is not None,
            }
        )

    return pd.DataFrame(rows)


def _least_squares_doa_position(
    anchors: np.ndarray,
    bearings_rad: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> Tuple[Optional[np.ndarray], Optional[float], int]:
    valid = np.isfinite(anchors).all(axis=1) & np.isfinite(bearings_rad)
    if weights is not None:
        valid &= np.isfinite(weights) & (weights > 0)

    anchors = anchors[valid]
    bearings_rad = bearings_rad[valid]
    resolved_weights = weights[valid] if weights is not None else None
    anchor_count = len(anchors)
    if anchor_count < 2:
        return None, None, anchor_count

    normals = np.column_stack([-np.sin(bearings_rad), np.cos(bearings_rad)])
    b_vector = np.sum(normals * anchors, axis=1)
    a_matrix = normals

    if resolved_weights is not None:
        sqrt_weights = np.sqrt(resolved_weights)
        a_matrix = a_matrix * sqrt_weights[:, np.newaxis]
        b_vector = b_vector * sqrt_weights

    try:
        solution, _, rank, _ = np.linalg.lstsq(a_matrix, b_vector, rcond=None)
    except np.linalg.LinAlgError:
        return None, None, anchor_count

    if rank < 2 or not np.isfinite(solution).all():
        return None, None, anchor_count

    line_residuals = normals @ solution - np.sum(normals * anchors, axis=1)
    rmse = float(np.sqrt(np.mean(line_residuals**2)))
    return solution, rmse, anchor_count


def estimate_doa_positions(doa_df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        doa_df,
        [
            "scenario_id",
            "target_id",
            "antenna_x",
            "antenna_y",
            "observed_bearing_rad",
        ],
        "links_doa.parquet",
    )

    rows = []
    for (scenario_id, target_id), group in doa_df.groupby(KEY_COLUMNS, sort=True):
        anchors = group[["antenna_x", "antenna_y"]].astype(float).to_numpy()
        bearings = group["observed_bearing_rad"].astype(float).to_numpy()
        sigma = (
            group["doa_noise_sigma_rad"].astype(float).to_numpy()
            if "doa_noise_sigma_rad" in group.columns
            else np.ones(len(group), dtype=float)
        )
        weights = 1.0 / np.maximum(sigma, EPSILON) ** 2
        position, residual_rmse, anchor_count = _least_squares_doa_position(
            anchors,
            bearings,
            weights=weights,
        )
        rows.append(
            {
                "scenario_id": scenario_id,
                "target_id": target_id,
                "doa_anchor_count": anchor_count,
                "doa_est_x": float(position[0]) if position is not None else np.nan,
                "doa_est_y": float(position[1]) if position is not None else np.nan,
                "doa_residual_rmse_m": residual_rmse,
                "doa_success": position is not None,
            }
        )

    return pd.DataFrame(rows)


def build_position_estimates(data_dir: Union[str, Path]) -> pd.DataFrame:
    links_df = _read_table(
        data_dir,
        "links",
        [
            "scenario_id",
            "target_id",
            "target_x",
            "target_y",
            "antenna_id",
            "distance_m",
        ],
    )
    rssi_df = _read_table(data_dir, "links_rssi", ["scenario_id", "target_id"])
    tdoa_df = _read_table(data_dir, "links_tdoa", ["scenario_id", "target_id"])
    doa_df = _read_table(data_dir, "links_doa", ["scenario_id", "target_id"])

    estimates_df = _target_truth_from_links(links_df)
    estimates_df = estimates_df.merge(
        estimate_rssi_positions(rssi_df),
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    estimates_df = estimates_df.merge(
        estimate_tdoa_positions(tdoa_df),
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    estimates_df = estimates_df.merge(
        estimate_doa_positions(doa_df),
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )

    for method in ("rssi", "tdoa", "doa"):
        estimates_df[f"{method}_error_m"] = [
            _position_error_m(est_x, est_y, target_x, target_y)
            for est_x, est_y, target_x, target_y in zip(
                estimates_df[f"{method}_est_x"],
                estimates_df[f"{method}_est_y"],
                estimates_df["target_x"],
                estimates_df["target_y"],
            )
        ]

    ordered_cols = [
        "scenario_id",
        "target_id",
        "target_label",
        "target_x",
        "target_y",
        "target_cell_id",
        "target_row_idx",
        "target_col_idx",
        "target_space_type",
        "target_room_id",
        "target_room_type",
        "target_patio_id",
        "rssi_est_x",
        "rssi_est_y",
        "rssi_error_m",
        "rssi_residual_rmse_m",
        "rssi_anchor_count",
        "rssi_success",
        "tdoa_est_x",
        "tdoa_est_y",
        "tdoa_error_m",
        "tdoa_residual_rmse_m",
        "tdoa_anchor_count",
        "tdoa_success",
        "doa_est_x",
        "doa_est_y",
        "doa_error_m",
        "doa_residual_rmse_m",
        "doa_anchor_count",
        "doa_success",
    ]
    return estimates_df[[column for column in ordered_cols if column in estimates_df.columns]].copy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate target positions from RSSI, TDOA, and DOA measurement tables."
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
        help="Output parquet path. Default: <data-dir>/position_estimates.parquet",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_path = Path(args.output) if args.output else data_dir / "position_estimates.parquet"
    position_estimates_df = build_position_estimates(data_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    position_estimates_df.to_parquet(output_path, index=False)

    print(f"Wrote {len(position_estimates_df)} position estimate rows to {output_path}")
    for method in ("rssi", "tdoa", "doa"):
        success_count = int(position_estimates_df[f"{method}_success"].fillna(False).sum())
        print(f"{method.upper()} successful estimates: {success_count}/{len(position_estimates_df)}")


if __name__ == "__main__":
    main()
