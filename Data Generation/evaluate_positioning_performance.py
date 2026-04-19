"""
POSITIONING PERFORMANCE EVALUATION
----------------------------------

Compare RSSI, TDOA, DOA/AOA, and optional ML positioning performance using
2D localisation error in metres.

Input tables:
- ml_dataset.parquet if available, otherwise position_estimates.parquet
- optional ML predictions file with scenario_id, target_id, and ML estimate
  coordinates

  NOTE: This will be updated once the ML pipeline and prediction output formats are done.

Outputs:
- method_performance_summary.csv
- condition_performance_summary.csv
- method_error_distribution.png
- method_error_cdf.png
- ml_vs_baseline_improvement.csv when ML predictions are available
- paired_method_comparisons.csv when ML predictions are available

Usage examples:
python3 "Data Generation/evaluate_positioning_performance.py" \
  --data-dir "Data Generation/generated_network_scenarios"

python3 "Data Generation/evaluate_positioning_performance.py" \
  --data-dir "Data Generation/generated_network_scenarios" \
  --predictions-path "Data Generation/generated_network_scenarios/ml_predictions.parquet"

@author: Giuliana Emberson
@date: 7th of May 2026

"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd


KEY_COLUMNS = ["scenario_id", "target_id"]
CONVENTIONAL_METHODS = ("rssi", "tdoa", "doa")
METHOD_LABELS = {
    "rssi": "RSSI",
    "tdoa": "TDOA",
    "doa": "DOA",
    "ml": "ML",
}
ML_ESTIMATE_COLUMN_CANDIDATES = [
    ("ml_est_x", "ml_est_y"),
    ("predicted_x", "predicted_y"),
    ("prediction_x", "prediction_y"),
    ("pred_x", "pred_y"),
    ("x_pred", "y_pred"),
]


def _get_pyplot():
    import os

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex_mpl")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

    import matplotlib.pyplot as plt

    return plt


def _require_columns(df: pd.DataFrame, required: Iterable[str], table_name: str) -> None:
    missing = set(required).difference(df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"{table_name} is missing required columns: {missing_list}")


def _read_table(path: Union[str, Path], required: Iterable[str], table_name: str) -> pd.DataFrame:
    resolved_path = Path(path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Missing required table: {resolved_path}")

    if resolved_path.suffix.lower() == ".csv":
        df = pd.read_csv(resolved_path)
    elif resolved_path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(resolved_path)
    else:
        raise ValueError(
            f"Unsupported file type for {resolved_path}. Use .parquet or .csv."
        )

    _require_columns(df, required, table_name)
    return df


def _input_dataset_path(data_dir: Union[str, Path]) -> Path:
    data_path = Path(data_dir)
    ml_dataset_path = data_path / "ml_dataset.parquet"
    if ml_dataset_path.exists():
        return ml_dataset_path

    estimates_path = data_path / "position_estimates.parquet"
    if estimates_path.exists():
        return estimates_path

    raise FileNotFoundError(
        f"Could not find {ml_dataset_path} or {estimates_path}. "
        "Run position estimation and ML dataset generation first."
    )


def _read_base_dataset(data_dir: Union[str, Path]) -> pd.DataFrame:
    path = _input_dataset_path(data_dir)
    return _read_table(
        path,
        ["scenario_id", "target_id", "target_x", "target_y"],
        path.name,
    )


def _resolve_ml_estimate_columns(predictions_df: pd.DataFrame) -> Optional[Tuple[str, str]]:
    for x_col, y_col in ML_ESTIMATE_COLUMN_CANDIDATES:
        if x_col in predictions_df.columns and y_col in predictions_df.columns:
            return x_col, y_col
    return None


def _merge_ml_predictions(
    base_df: pd.DataFrame,
    predictions_path: Optional[Union[str, Path]],
) -> pd.DataFrame:
    dataset_df = base_df.copy()
    if predictions_path is None:
        if "ml_error_m" not in dataset_df.columns and {"ml_est_x", "ml_est_y"}.issubset(dataset_df.columns):
            dataset_df["ml_error_m"] = _euclidean_error(
                dataset_df["ml_est_x"],
                dataset_df["ml_est_y"],
                dataset_df["target_x"],
                dataset_df["target_y"],
            )
        if "ml_error_m" in dataset_df.columns and "ml_success" not in dataset_df.columns:
            dataset_df["ml_success"] = dataset_df["ml_error_m"].notna()
        return dataset_df

    predictions_df = _read_table(
        predictions_path,
        KEY_COLUMNS,
        Path(predictions_path).name,
    )
    estimate_cols = _resolve_ml_estimate_columns(predictions_df)
    prediction_cols = [*KEY_COLUMNS]

    if estimate_cols is not None:
        x_col, y_col = estimate_cols
        prediction_cols.extend([x_col, y_col])
    elif "ml_error_m" not in predictions_df.columns:
        raise ValueError(
            "ML predictions must include ml_est_x/ml_est_y, predicted_x/predicted_y, "
            "prediction_x/prediction_y, pred_x/pred_y, x_pred/y_pred, or ml_error_m."
        )

    for optional_col in ("ml_error_m", "ml_success"):
        if optional_col in predictions_df.columns:
            prediction_cols.append(optional_col)

    prediction_subset_df = predictions_df[prediction_cols].drop_duplicates(KEY_COLUMNS).copy()
    duplicates = prediction_subset_df.duplicated(KEY_COLUMNS, keep=False)
    if duplicates.any():
        duplicate_count = int(duplicates.sum())
        raise ValueError(
            f"ML predictions contain {duplicate_count} duplicate scenario-target rows."
        )

    if estimate_cols is not None and estimate_cols != ("ml_est_x", "ml_est_y"):
        prediction_subset_df = prediction_subset_df.rename(
            columns={estimate_cols[0]: "ml_est_x", estimate_cols[1]: "ml_est_y"}
        )

    drop_cols = [
        column
        for column in ("ml_est_x", "ml_est_y", "ml_error_m", "ml_success")
        if column in dataset_df.columns
    ]
    if drop_cols:
        dataset_df = dataset_df.drop(columns=drop_cols)

    dataset_df = dataset_df.merge(
        prediction_subset_df,
        on=KEY_COLUMNS,
        how="inner",
        validate="one_to_one",
    )

    if "ml_error_m" not in dataset_df.columns:
        dataset_df["ml_error_m"] = _euclidean_error(
            dataset_df["ml_est_x"],
            dataset_df["ml_est_y"],
            dataset_df["target_x"],
            dataset_df["target_y"],
        )
    if "ml_success" not in dataset_df.columns:
        dataset_df["ml_success"] = dataset_df["ml_error_m"].notna()

    return dataset_df


def _euclidean_error(
    est_x: Union[pd.Series, Sequence[float]],
    est_y: Union[pd.Series, Sequence[float]],
    target_x: Union[pd.Series, Sequence[float]],
    target_y: Union[pd.Series, Sequence[float]],
) -> pd.Series:
    est_x_s = pd.Series(est_x, copy=False).astype(float)
    est_y_s = pd.Series(est_y, copy=False).astype(float)
    target_x_s = pd.Series(target_x, copy=False).astype(float)
    target_y_s = pd.Series(target_y, copy=False).astype(float)
    return np.hypot(est_x_s - target_x_s, est_y_s - target_y_s)


def _available_methods(df: pd.DataFrame) -> List[str]:
    methods = []
    for method in (*CONVENTIONAL_METHODS, "ml"):
        if f"{method}_error_m" in df.columns:
            methods.append(method)
    return methods


def _method_success_mask(df: pd.DataFrame, method: str) -> pd.Series:
    error_col = f"{method}_error_m"
    success_col = f"{method}_success"
    mask = df[error_col].notna() & np.isfinite(df[error_col].astype(float))
    if success_col in df.columns:
        mask &= df[success_col].fillna(False).astype(bool)
    return mask


def _rmse(errors: pd.Series) -> float:
    values = errors.astype(float).to_numpy()
    return float(np.sqrt(np.mean(values**2))) if len(values) else float("nan")


def _method_summary(df: pd.DataFrame, methods: Sequence[str]) -> pd.DataFrame:
    rows = []
    total_rows = len(df)
    for method in methods:
        error_col = f"{method}_error_m"
        success_mask = _method_success_mask(df, method)
        errors = df.loc[success_mask, error_col].astype(float)
        rows.append(
            {
                "method": METHOD_LABELS.get(method, method.upper()),
                "row_count": total_rows,
                "successful_count": int(success_mask.sum()),
                "success_rate": float(success_mask.mean()) if total_rows else float("nan"),
                "mean_error_m": float(errors.mean()) if len(errors) else float("nan"),
                "median_error_m": float(errors.median()) if len(errors) else float("nan"),
                "rmse_error_m": _rmse(errors),
                "std_error_m": float(errors.std()) if len(errors) > 1 else 0.0,
                "p75_error_m": float(errors.quantile(0.75)) if len(errors) else float("nan"),
                "p90_error_m": float(errors.quantile(0.90)) if len(errors) else float("nan"),
                "p95_error_m": float(errors.quantile(0.95)) if len(errors) else float("nan"),
                "max_error_m": float(errors.max()) if len(errors) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _condition_columns(df: pd.DataFrame) -> List[str]:
    condition_cols = []
    if "env_type" in df.columns:
        condition_cols.append("env_type")
    if "obstruction_severity" in df.columns:
        condition_cols.append("obstruction_severity")
    return condition_cols


def _add_obstruction_severity(df: pd.DataFrame) -> pd.DataFrame:
    result_df = df.copy()
    source_col = None
    if "link_total_blocker_mean" in result_df.columns:
        source_col = "link_total_blocker_mean"
    elif "link_nlos_count" in result_df.columns:
        source_col = "link_nlos_count"

    if source_col is None:
        return result_df

    values = result_df[source_col].fillna(0).astype(float)
    result_df["obstruction_severity"] = "none"
    nonzero = values > 0
    if nonzero.any():
        median_nonzero = float(values[nonzero].median())
        result_df.loc[nonzero & (values <= median_nonzero), "obstruction_severity"] = "low"
        result_df.loc[nonzero & (values > median_nonzero), "obstruction_severity"] = "high"
    return result_df


def _condition_summary(df: pd.DataFrame, methods: Sequence[str]) -> pd.DataFrame:
    rows = []
    for condition_col in _condition_columns(df):
        for condition_value, condition_df in df.groupby(condition_col, dropna=False, sort=True):
            summary_df = _method_summary(condition_df, methods)
            summary_df.insert(0, "condition_value", condition_value)
            summary_df.insert(0, "condition", condition_col)
            rows.append(summary_df)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _long_error_df(df: pd.DataFrame, methods: Sequence[str]) -> pd.DataFrame:
    rows = []
    id_cols = [column for column in [*KEY_COLUMNS, "env_type", "obstruction_severity"] if column in df.columns]
    for method in methods:
        error_col = f"{method}_error_m"
        success_mask = _method_success_mask(df, method)
        method_df = df.loc[success_mask, [*id_cols, error_col]].copy()
        method_df = method_df.rename(columns={error_col: "error_m"})
        method_df["method"] = METHOD_LABELS.get(method, method.upper())
        rows.append(method_df)
    if not rows:
        return pd.DataFrame(columns=[*id_cols, "error_m", "method"])
    return pd.concat(rows, ignore_index=True)


def _paired_bootstrap_ci(
    baseline_errors: np.ndarray,
    ml_errors: np.ndarray,
    *,
    statistic: str,
    rng: np.random.Generator,
    samples: int,
) -> Tuple[float, float]:
    if len(baseline_errors) == 0:
        return float("nan"), float("nan")

    differences = baseline_errors - ml_errors
    if len(differences) == 1:
        value = float(differences[0])
        return value, value

    if statistic not in {"mean", "median"}:
        raise ValueError(f"Unsupported bootstrap statistic: {statistic}")

    max_values_per_chunk = 2_000_000
    chunk_size = max(1, min(samples, max_values_per_chunk // len(differences)))
    values = []
    remaining = samples
    while remaining > 0:
        current_chunk = min(chunk_size, remaining)
        indices = rng.integers(0, len(differences), size=(current_chunk, len(differences)))
        resampled = differences[indices]
        if statistic == "median":
            values.append(np.median(resampled, axis=1))
        else:
            values.append(np.mean(resampled, axis=1))
        remaining -= current_chunk

    bootstrap_values = np.concatenate(values)
    return (
        float(np.percentile(bootstrap_values, 2.5)),
        float(np.percentile(bootstrap_values, 97.5)),
    )


def _wilcoxon_p_value(baseline_errors: np.ndarray, ml_errors: np.ndarray) -> float:
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return float("nan")

    differences = baseline_errors - ml_errors
    if len(differences) == 0 or np.allclose(differences, 0.0):
        return float("nan")

    try:
        return float(wilcoxon(baseline_errors, ml_errors).pvalue)
    except ValueError:
        return float("nan")


def _ml_comparisons(
    df: pd.DataFrame,
    methods: Sequence[str],
    *,
    bootstrap_samples: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if "ml" not in methods:
        return pd.DataFrame(), pd.DataFrame()

    rng = np.random.default_rng(seed)
    improvement_rows = []
    comparison_rows = []

    for baseline in CONVENTIONAL_METHODS:
        if baseline not in methods:
            continue

        baseline_mask = _method_success_mask(df, baseline)
        ml_mask = _method_success_mask(df, "ml")
        paired_df = df.loc[baseline_mask & ml_mask, [f"{baseline}_error_m", "ml_error_m"]].copy()
        paired_df = paired_df.dropna()
        if paired_df.empty:
            continue

        baseline_errors = paired_df[f"{baseline}_error_m"].astype(float).to_numpy()
        ml_errors = paired_df["ml_error_m"].astype(float).to_numpy()
        differences = baseline_errors - ml_errors
        valid_improvement = baseline_errors > 0
        improvements = 100.0 * differences[valid_improvement] / baseline_errors[valid_improvement]
        mean_ci = _paired_bootstrap_ci(
            baseline_errors,
            ml_errors,
            statistic="mean",
            rng=rng,
            samples=bootstrap_samples,
        )
        median_ci = _paired_bootstrap_ci(
            baseline_errors,
            ml_errors,
            statistic="median",
            rng=rng,
            samples=bootstrap_samples,
        )

        comparison_rows.append(
            {
                "comparison": f"ML vs {METHOD_LABELS[baseline]}",
                "paired_count": len(paired_df),
                "baseline_mean_error_m": float(np.mean(baseline_errors)),
                "ml_mean_error_m": float(np.mean(ml_errors)),
                "mean_error_reduction_m": float(np.mean(differences)),
                "mean_error_reduction_ci95_low_m": mean_ci[0],
                "mean_error_reduction_ci95_high_m": mean_ci[1],
                "median_error_reduction_m": float(np.median(differences)),
                "median_error_reduction_ci95_low_m": median_ci[0],
                "median_error_reduction_ci95_high_m": median_ci[1],
                "wilcoxon_p_value": _wilcoxon_p_value(baseline_errors, ml_errors),
            }
        )
        improvement_rows.append(
            {
                "baseline_method": METHOD_LABELS[baseline],
                "paired_count": len(paired_df),
                "mean_improvement_percent": float(np.mean(improvements)) if len(improvements) else float("nan"),
                "median_improvement_percent": float(np.median(improvements)) if len(improvements) else float("nan"),
                "p25_improvement_percent": float(np.percentile(improvements, 25)) if len(improvements) else float("nan"),
                "p75_improvement_percent": float(np.percentile(improvements, 75)) if len(improvements) else float("nan"),
            }
        )

    conventional_errors = [
        f"{method}_error_m"
        for method in CONVENTIONAL_METHODS
        if method in methods
    ]
    if conventional_errors:
        paired_mask = _method_success_mask(df, "ml")
        for method in CONVENTIONAL_METHODS:
            if method in methods:
                paired_mask &= _method_success_mask(df, method)
        best_df = df.loc[paired_mask, [*conventional_errors, "ml_error_m"]].dropna().copy()
        if not best_df.empty:
            best_errors = best_df[conventional_errors].min(axis=1).astype(float).to_numpy()
            ml_errors = best_df["ml_error_m"].astype(float).to_numpy()
            differences = best_errors - ml_errors
            valid_improvement = best_errors > 0
            improvements = 100.0 * differences[valid_improvement] / best_errors[valid_improvement]
            mean_ci = _paired_bootstrap_ci(
                best_errors,
                ml_errors,
                statistic="mean",
                rng=rng,
                samples=bootstrap_samples,
            )
            median_ci = _paired_bootstrap_ci(
                best_errors,
                ml_errors,
                statistic="median",
                rng=rng,
                samples=bootstrap_samples,
            )
            comparison_rows.append(
                {
                    "comparison": "ML vs Best Conventional",
                    "paired_count": len(best_df),
                    "baseline_mean_error_m": float(np.mean(best_errors)),
                    "ml_mean_error_m": float(np.mean(ml_errors)),
                    "mean_error_reduction_m": float(np.mean(differences)),
                    "mean_error_reduction_ci95_low_m": mean_ci[0],
                    "mean_error_reduction_ci95_high_m": mean_ci[1],
                    "median_error_reduction_m": float(np.median(differences)),
                    "median_error_reduction_ci95_low_m": median_ci[0],
                    "median_error_reduction_ci95_high_m": median_ci[1],
                    "wilcoxon_p_value": _wilcoxon_p_value(best_errors, ml_errors),
                }
            )
            improvement_rows.append(
                {
                    "baseline_method": "Best Conventional",
                    "paired_count": len(best_df),
                    "mean_improvement_percent": float(np.mean(improvements)) if len(improvements) else float("nan"),
                    "median_improvement_percent": float(np.median(improvements)) if len(improvements) else float("nan"),
                    "p25_improvement_percent": float(np.percentile(improvements, 25)) if len(improvements) else float("nan"),
                    "p75_improvement_percent": float(np.percentile(improvements, 75)) if len(improvements) else float("nan"),
                }
            )

    return pd.DataFrame(improvement_rows), pd.DataFrame(comparison_rows)


def _plot_error_distribution(long_df: pd.DataFrame, output_path: Path) -> None:
    if long_df.empty:
        return

    plt = _get_pyplot()
    methods = list(long_df["method"].drop_duplicates())
    data = [
        long_df.loc[long_df["method"] == method, "error_m"].astype(float).to_numpy()
        for method in methods
    ]
    all_errors = long_df["error_m"].astype(float)
    upper = float(all_errors.quantile(0.99)) if len(all_errors) else 1.0
    upper = max(upper, 1.0)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot(data, tick_labels=methods, showfliers=False)
    ax.set_ylabel("2D localisation error (m)")
    ax.set_title("Positioning Error Distribution")
    ax.set_ylim(bottom=0, top=upper * 1.05)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_error_cdf(long_df: pd.DataFrame, output_path: Path) -> None:
    if long_df.empty:
        return

    plt = _get_pyplot()
    fig, ax = plt.subplots(figsize=(9, 5))
    for method, method_df in long_df.groupby("method", sort=False):
        errors = np.sort(method_df["error_m"].astype(float).to_numpy())
        if len(errors) == 0:
            continue
        cdf = np.arange(1, len(errors) + 1) / len(errors)
        ax.plot(errors, cdf, label=method)

    all_errors = long_df["error_m"].astype(float)
    upper = float(all_errors.quantile(0.99)) if len(all_errors) else 1.0
    ax.set_xlim(left=0, right=max(upper, 1.0))
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("2D localisation error (m)")
    ax.set_ylabel("Cumulative proportion of targets")
    ax.set_title("Positioning Error CDF")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_evaluation_outputs(
    data_dir: Union[str, Path],
    *,
    predictions_path: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    bootstrap_samples: int = 2000,
    seed: int = 42,
) -> Dict[str, Path]:
    dataset_df = _read_base_dataset(data_dir)
    dataset_df = _merge_ml_predictions(dataset_df, predictions_path)
    dataset_df = _add_obstruction_severity(dataset_df)
    methods = _available_methods(dataset_df)
    if not methods:
        raise ValueError("No method error columns were found for evaluation.")

    resolved_output_dir = (
        Path(output_dir)
        if output_dir is not None
        else Path(data_dir) / "evaluation"
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    summary_df = _method_summary(dataset_df, methods)
    condition_summary_df = _condition_summary(dataset_df, methods)
    long_df = _long_error_df(dataset_df, methods)
    improvement_df, paired_df = _ml_comparisons(
        dataset_df,
        methods,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )

    paths = {
        "method_summary": resolved_output_dir / "method_performance_summary.csv",
        "condition_summary": resolved_output_dir / "condition_performance_summary.csv",
        "error_distribution": resolved_output_dir / "method_error_distribution.png",
        "error_cdf": resolved_output_dir / "method_error_cdf.png",
    }

    summary_df.to_csv(paths["method_summary"], index=False)
    condition_summary_df.to_csv(paths["condition_summary"], index=False)
    _plot_error_distribution(long_df, paths["error_distribution"])
    _plot_error_cdf(long_df, paths["error_cdf"])

    if not improvement_df.empty:
        paths["ml_improvement"] = resolved_output_dir / "ml_vs_baseline_improvement.csv"
        improvement_df.to_csv(paths["ml_improvement"], index=False)
    if not paired_df.empty:
        paths["paired_comparisons"] = resolved_output_dir / "paired_method_comparisons.csv"
        paired_df.to_csv(paths["paired_comparisons"], index=False)

    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate RSSI, TDOA, DOA/AOA, and optional ML positioning performance."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="generated_network_scenarios",
        help="Directory containing position_estimates.parquet or ml_dataset.parquet.",
    )
    parser.add_argument(
        "--predictions-path",
        type=str,
        default=None,
        help="Optional CSV or parquet file containing ML predictions.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for CSV summaries and plots. Default: <data-dir>/evaluation",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=2000,
        help="Number of paired bootstrap samples for ML comparison confidence intervals.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for bootstrap resampling.",
    )
    args = parser.parse_args()

    if args.bootstrap_samples < 1:
        raise ValueError("--bootstrap-samples must be >= 1.")

    paths = build_evaluation_outputs(
        args.data_dir,
        predictions_path=args.predictions_path,
        output_dir=args.output_dir,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    print("Wrote positioning performance outputs:")
    for label, path in paths.items():
        print(f"- {label}: {path}")


if __name__ == "__main__":
    main()
