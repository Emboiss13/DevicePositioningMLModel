import argparse
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd
from general_structure import NetworkScenario


"""
Generate one or more random network scenarios and print a short summary for each.

Usage examples
--------------
python3 run_scenarios.py --count 5
python3 run_scenarios.py --count 10 --seed 2
"""

def summarize_scenario(idx: int, scenario: NetworkScenario) -> None:
    scenario_dict = scenario.to_dict()
    print(f"Scenario {idx}")
    print(
        "  env:",
        scenario_dict["environment"]["env_type"],
        "area:",
        round(scenario_dict["environment"]["grid_area"], 2),
    )
    print(
        "  devices:", len(scenario_dict["devices"]),
        "obstacles:", len(scenario_dict["obstacles"]),
        "channels:", len(scenario_dict["channels"]),
    )
    print("  target_selected:", scenario_dict["target_selected"])
    print()


def scenario_to_rows(
    scenario_id: str,
    seed: int,
    scenario: NetworkScenario,
) -> Dict[str, List[Dict[str, Any]]]:
    """Flatten a scenario into row dictionaries for each table."""
    dict = scenario.to_dict()

    summary_row = {
        "scenario_id": scenario_id,
        "seed": seed,
        "label": dict["label"],
        "env_type": dict["environment"]["env_type"],
        "grid_area": dict["environment"]["grid_area"],
        "width": dict["environment"]["width"],
        "height": dict["environment"]["height"],
        "target_selected": dict["target_selected"],
        "device_count": len(dict["devices"]),
        "obstacle_count": len(dict["obstacles"]),
        "channel_count": len(dict["channels"]),
    }

    device_rows = []
    for device in dict["devices"]:
        device_rows.append(
            {
                "scenario_id": scenario_id,
                "seed": seed,
                "device_id": device["device_id"],
                "device_type": device["device_type"],
                "is_target": device["is_target"],
                "x": device["position"][0],
                "y": device["position"][1],
            }
        )

    obstacle_rows = []
    for obstacle in dict["obstacles"]:
        p0 = obstacle["position_X_Y"]
        p1 = obstacle.get("position_X1_Y1")
        obstacle_rows.append(
            {
                "scenario_id": scenario_id,
                "seed": seed,
                "obstacle_id": obstacle["obstacle_id"],
                "obstacle_type": obstacle["obstacle_type"],
                "x0": p0[0],
                "y0": p0[1],
                "x1": p1[0] if p1 else None,
                "y1": p1[1] if p1 else None,
                "radius": obstacle.get("radius"),
                "length": obstacle.get("length"),
                "width": obstacle.get("width"),
                "area": obstacle["area"],
            }
        )

    channel_rows = []
    for channel in dict["channels"]:
        channel_rows.append(
            {
                "scenario_id": scenario_id,
                "seed": seed,
                "device_a_id": channel["device_a_id"],
                "device_b_id": channel["device_b_id"],
                "distance_m": channel["distance_m"],
                "freq_mhz": channel["freq_mhz"],
                "blocking_obstacles": channel["blocking_obstacles"],
            }
        )

    return {
        "summary": [summary_row],
        "devices": device_rows,
        "obstacles": obstacle_rows,
        "channels": channel_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate random network scenarios")
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of scenarios to generate (default: 1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional base seed for reproducibility; each scenario offsets this seed",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="general_data",
        help="Directory where Parquet files will be written (default: general_data)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="If set, skip writing Parquet files (only print summaries).",
    )
    args = parser.parse_args()

    base_seed = args.seed
    out_dir = Path(args.output_dir)

    # Collect rows for each table across all scenarios
    summary_rows: List[Dict[str, Any]] = []
    device_rows: List[Dict[str, Any]] = []
    obstacle_rows: List[Dict[str, Any]] = []
    channel_rows: List[Dict[str, Any]] = []

    for iteration in range(1, args.count + 1):
        # Vary the seed per scenario so each run is different even with a base seed
        scenario_seed = None if base_seed is None else base_seed + (iteration - 1)
        network_scenario = NetworkScenario.generate_random(seed=scenario_seed)
        summarize_scenario(iteration, network_scenario)
        if not args.no_save:
            scenario_id = f"scenario_{iteration:04d}"
            rows = scenario_to_rows(scenario_id, scenario_seed, network_scenario)
            summary_rows.extend(rows["summary"])
            device_rows.extend(rows["devices"])
            obstacle_rows.extend(rows["obstacles"])
            channel_rows.extend(rows["channels"])

    if not args.no_save:
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(summary_rows).to_parquet(out_dir / "summary.parquet", index=False)
        pd.DataFrame(device_rows).to_parquet(out_dir / "devices.parquet", index=False)
        pd.DataFrame(obstacle_rows).to_parquet(out_dir / "obstacles.parquet", index=False)
        pd.DataFrame(channel_rows).to_parquet(out_dir / "channels.parquet", index=False)


if __name__ == "__main__":
    main()