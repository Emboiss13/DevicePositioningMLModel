import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

import matplotlib
matplotlib.use("Agg")  # headless / non-GUI
import matplotlib.pyplot as plt
import pandas as pd
from general_envs import NetworkScenario


"""
Generate one or more random network scenarios and print a short summary for each.

USAGE EXAMPLES:
--------------
python3 create_network_envs.py --count 5
python3 create_network_envs.py --count 10 --seed 2
python3 create_network_envs.py --count 1 --output-dir generated_network_scenarios --plot
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


def plot_scenario(scenario_id: str, scenario: NetworkScenario, out_dir: Path) -> Optional[str]:
    """
    Render the scenario (devices, obstacles, channels) and save to PNG.
    Returns the relative path to the saved image, or None on failure.
    """
    try:
        fig, ax = plt.subplots(figsize=(8, 8))
        d = scenario.to_dict()

        # Environment bounds
        xmin, xmax = d["environment"]["x_domain"]
        ymin, ymax = d["environment"]["y_range"]
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{scenario_id} - {d['environment']['env_type']}")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")

        # Obstacles
        for ob in d["obstacles"]:
            ob_type = ob["obstacle_type"]
            color = "tab:red" if ob_type in ("human", "stairs") else "tab:gray"

            if ob_type == "human" and ob.get("radius") is not None:
                circle = plt.Circle(ob["position_X_Y"], ob["radius"], color=color, alpha=0.35, ec="k")
                ax.add_artist(circle)
            else:
                p0 = ob["position_X_Y"]
                p1 = ob.get("position_X1_Y1")
                if p1:
                    width = p1[0] - p0[0]
                    height = p1[1] - p0[1]
                    rect = plt.Rectangle(p0, width, height, color=color, alpha=0.35, ec="k")
                    ax.add_patch(rect)

        # Devices
        for dev in d["devices"]:
            x, y = dev["position"]
            if dev["device_type"] == "target_endpoint":
                ax.scatter(x, y, color="gold", edgecolors="k", s=80, marker="*", label="target")
            else:
                ax.scatter(x, y, color="tab:blue", edgecolors="k", s=50, marker="^", label="antenna")

        # Channels
        show_los_label = True
        env_type = d["environment"]["env_type"]
        for ch in d["channels"]:
            xa, ya = ch["device_a_position"]
            xb, yb = ch["device_b_position"]

            is_free_los = env_type in ("indoor_LOS", "outdoor") and ch.get("blocking_obstacles", 0) == 0
            if is_free_los:
                ax.plot([xa, xb], [ya, yb], color="red", alpha=0.7, linewidth=2,
                        label="LOS free" if show_los_label else None)
                show_los_label = False
            else:
                ax.plot([xa, xb], [ya, yb], color="gray", alpha=0.4, linewidth=1)

        # Deduplicate legend
        handles, labels = ax.get_legend_handles_labels()
        uniq = dict(zip(labels, handles))
        if uniq:
            ax.legend(uniq.values(), uniq.keys(), loc="upper right")

        plots_dir = out_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        png_path = plots_dir / f"{scenario_id}.png"
        fig.tight_layout()
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        return str(png_path.relative_to(out_dir))
    except Exception:
        plt.close("all")
        return None


def scenario_to_rows(
    scenario_id: str,
    seed: int,
    scenario: NetworkScenario,
    plot_path: Optional[str],
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
        "plot_path": plot_path,
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
        default="generated_network_scenarios",
        help="Directory where Parquet files will be written (default: generated_network_scenarios)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="If set, skip writing Parquet files (only print summaries).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Render and save a PNG per scenario and store its path in summary.parquet.",
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
            plot_path = None
            if args.plot:
                plot_path = plot_scenario(scenario_id, network_scenario, out_dir)

            rows = scenario_to_rows(scenario_id, scenario_seed, network_scenario, plot_path)
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