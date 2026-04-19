## Dependencies

- matplotlib (plotting)
- fastparquet (parquet columnar data storage format)
- pyarrow (parquet columnar data storage format)
- renovation (floorplan generation)

## Pipeline

The data-generation stage is split into six steps:

1. Generate network scenarios, floor plans, targets, and shared antenna-target
   links:

```bash
python3 "Data Generation/create_network_envs.py" --count 1 --seed 7 --output-dir "generated_network_scenarios"
```

2. Generate method-specific RSSI, TDOA, and DOA/AOA measurements:

```bash
python3 "Data Generation/RSSI/RSSI_envs.py" --data-dir "generated_network_scenarios" --seed 7
python3 "Data Generation/TDOA/TDOA_envs.py" --data-dir "generated_network_scenarios" --seed 7
python3 "Data Generation/DOA/DOA_envs.py" --data-dir "generated_network_scenarios" --seed 7
```

3. Estimate target positions using conventional RSSI, TDOA, and DOA/AOA
   methods:

```bash
python3 "Data Generation/position_estimation.py" --data-dir "generated_network_scenarios"
```

4. Build the labelled ML dataset:

```bash
python3 "Data Generation/create_ml_dataset.py" --data-dir "generated_network_scenarios"
```

5. Validate that table counts, labels, measurements, estimates, and ML rows are
   consistent:

```bash
python3 "Data Generation/validate_generation_outputs.py" --data-dir "generated_network_scenarios"
```

6. Evaluate RSSI, TDOA, DOA/AOA, and optional ML positioning performance:

```bash
python3 "Data Generation/evaluate_positioning_performance.py" --data-dir "generated_network_scenarios"
```

## Output Tables

- `env_summary.parquet`: one row per scenario.
- `antennas.parquet`: antenna positions and coverage radius.
- `humans.parquet`: generated human obstacles.
- `floor_plan_elements.parquet`: wall, door, and window geometry.
- `grid_cells.parquet`: all generated target-grid cells.
- `targets.parquet`: valid target positions and ground-truth labels.
- `links.parquet`: one row per antenna-target pair with distance and LOS/NLOS
  geometry.
- `links_rssi.parquet`: RSSI/path-loss measurements per antenna-target link.
- `links_tdoa.parquet`: TDOA measurements per target and non-reference antenna.
- `links_doa.parquet`: DOA/AOA bearing measurements per antenna-target link.
- `position_estimates.parquet`: one row per target with conventional RSSI,
  TDOA, and DOA/AOA `(x, y)` estimates and error metrics.
- `ml_dataset.parquet`: one labelled row per target for the ML pipeline.
- `evaluation/`: CSV summaries and plots comparing positioning error across
  RSSI, TDOA, DOA/AOA, and optional ML predictions.

## Performance Evaluation

`evaluate_positioning_performance.py` reports 2D localisation error in metres:

```text
error_m = sqrt((estimated_x - target_x)^2 + (estimated_y - target_y)^2)
```

By default, the evaluator compares `rssi_error_m`, `tdoa_error_m`, and
`doa_error_m`. Once ML predictions are available, pass a CSV or parquet file
containing `scenario_id`, `target_id`, and either `ml_est_x`/`ml_est_y` or
`ml_error_m`:

```bash
python3 "Data Generation/evaluate_positioning_performance.py" \
  --data-dir "generated_network_scenarios" \
  --predictions-path "generated_network_scenarios/ml_predictions.parquet"
```

The evaluator writes:

- `method_performance_summary.csv`
- `condition_performance_summary.csv`
- `method_error_distribution.png`
- `method_error_cdf.png`
- `ml_vs_baseline_improvement.csv` when ML predictions are supplied.
- `paired_method_comparisons.csv` when ML predictions are supplied.

## Attenuation Scope

The current implementation uses floor-plan geometry to classify links as LOS or
NLOS and to count wall and human blockers. RSSI can apply uniform wall and human
attenuation through `--wall-loss-db` and `--human-loss-db`; by default these are
`0.0`, so no additional obstacle attenuation is applied. Material-specific
attenuation coefficients are not currently modelled.
