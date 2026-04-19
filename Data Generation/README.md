## Dependencies

- matplotlib (plotting)
- fastparquet (parquet columnar data storage format)
- pyarrow (parquet columnar data storage format)
- renovation (floorplan generation)

## Pipeline

The data-generation stage is split into five steps:

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

## Attenuation Scope

The current implementation uses floor-plan geometry to classify links as LOS or
NLOS and to count wall and human blockers. RSSI can apply uniform wall and human
attenuation through `--wall-loss-db` and `--human-loss-db`; by default these are
`0.0`, so no additional obstacle attenuation is applied. Material-specific
attenuation coefficients are not currently modelled.
