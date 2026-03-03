import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from network_scenario_factory import NetworkScenario

matplotlib.use("Agg")  # headless / non-GUI

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
        "  area:",
        round(scenario_dict["environment"]["grid_area"], 2),
        "width:",
        round(scenario_dict["environment"]["width"], 2),
        "height:",
        round(scenario_dict["environment"]["height"], 2),
    )
    print(
        "  antennas:", len(scenario_dict["antennas"]),
        "obstacles:", len(scenario_dict["obstacles"]),
    )
    print()


def plot_scenario(scenario_id: str, scenario: NetworkScenario, out_dir: Path) -> Optional[str]:
    """
    Render the scenario (antennas + obstacles) and save to PNG.
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
        ax.set_title(f"{scenario_id}")
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

        # Antennas + coverage
        for idx, antenna in enumerate(d["antennas"]):
            x, y = antenna["position"]
            radius = antenna["coverage_radius"]
            ax.scatter(
                x,
                y,
                color="tab:blue",
                edgecolors="k",
                s=50,
                marker="^",
                label="antenna" if idx == 0 else None,
            )
            cover = plt.Circle((x, y), radius, color="tab:blue", alpha=0.07, ec="tab:blue")
            ax.add_artist(cover)

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
    seed: Optional[int],
    scenario: NetworkScenario,
    plot_path: Optional[str],
) -> Dict[str, List[Dict[str, Any]]]:
    """Flatten a scenario into row dictionaries for each table."""
    scenario_dict = scenario.to_dict()

    env = scenario_dict["environment"]
    env_summary_row = {
        "scenario_id": scenario_id,
        "seed": seed,
        "grid_area": env["grid_area"],
        "width": env["width"],
        "height": env["height"],
        "x_domain_min": env["x_domain"][0],
        "x_domain_max": env["x_domain"][1],
        "y_range_min": env["y_range"][0],
        "y_range_max": env["y_range"][1],
        "antenna_count": len(scenario_dict["antennas"]),
        "obstacle_count": len(scenario_dict["obstacles"]),
        "plot_path": plot_path,
    }

    antenna_rows = []
    for antenna in scenario_dict["antennas"]:
        antenna_rows.append(
            {
                "scenario_id": scenario_id,
                "seed": seed,
                "antenna_id": antenna["antenna_id"],
                "antenna_label": antenna["antenna_label"],
                "x": antenna["position"][0],
                "y": antenna["position"][1],
                "coverage_radius": antenna["coverage_radius"],
            }
        )

    obstacle_rows = []
    for obstacle in scenario_dict["obstacles"]:
        p0 = obstacle["position_X_Y"]
        p1 = obstacle.get("position_X1_Y1")
        obstacle_rows.append(
            {
                "scenario_id": scenario_id,
                "seed": seed,
                "obstacle_id": obstacle["obstacle_id"],
                "obstacle_label": obstacle.get("obstacle_label"),
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

    return {
        "env_summary": [env_summary_row],
        "antennas": antenna_rows,
        "obstacles": obstacle_rows,
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
    env_summary_rows: List[Dict[str, Any]] = []
    antenna_rows: List[Dict[str, Any]] = []
    obstacle_rows: List[Dict[str, Any]] = []

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
            env_summary_rows.extend(rows["env_summary"])
            antenna_rows.extend(rows["antennas"])
            obstacle_rows.extend(rows["obstacles"])

    if not args.no_save:
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(env_summary_rows).to_parquet(out_dir / "env_summary.parquet", index=False)
        pd.DataFrame(antenna_rows).to_parquet(out_dir / "antennas.parquet", index=False)
        pd.DataFrame(obstacle_rows).to_parquet(out_dir / "obstacles.parquet", index=False)


if __name__ == "__main__":
    main()
