"""
Generate one or more random network scenarios/environments and print a short summary for each.

USAGE EXAMPLES:
--------------
python3 create_network_envs.py --count 5
python3 create_network_envs.py --count 10 --seed 2
python3 create_network_envs.py --count 1 --output-dir generated_network_scenarios --plot
python3 'Data Generation/create_network_envs.py' --count 1 --seed 7 --output-dir '/tmp/generated_network_scenarios_target_plot' --target-plot

python3 "Data Generation/create_network_envs.py" \
  --count 1 \
  --seed 7 \
  --output-dir "Data Generation/generated_network_scenarios" \
  --plot \
  --floorplan-plot

python3 "Data Generation/create_network_envs.py"  --count 100 --seed 13 --output-dir "Data Generation/generated_network_scenarios_with_plots" --plot --floorplan-plot

@author: Giuliana Emberson
@date: 7th of May 2026

"""

from __future__ import annotations
import argparse
import math
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional
import pandas as pd
from floor_plan_factory import GeneratedFloorPlan, generate_floor_plan_from_environment
from link_factory import build_link_rows
from network_scenario_factory import NetworkScenario

if TYPE_CHECKING:
    from matplotlib.axes import Axes

PLOT_DPI = 300
PLOT_LONG_EDGE_INCHES = 18.0
PLOT_MIN_SHORT_EDGE_INCHES = 10.0
PLOT_TARGET_PIXELS_PER_METER = 20.0
PLOT_MIN_LONG_EDGE_PIXELS = 5400
PLOT_MAX_LONG_EDGE_PIXELS = 9000


def _get_pyplot():
    import os

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex_mpl")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

    import matplotlib.pyplot as plt

    return plt

def summarize_scenario(
    scenario_id: str,
    scenario: NetworkScenario,
    floor_plan_summary: Optional[Dict[str, Any]] = None,
    target_summary: Optional[Dict[str, Any]] = None,
    timing_summary: Optional[Dict[str, float]] = None,
) -> None:
    scenario_dict = scenario.to_dict()
    print(scenario_id)
    print(
        "area:",
        round(scenario_dict["environment"]["area"], 2),
        "env_type:",
        scenario_dict["environment"]["env_type"],
        "width:",
        round(scenario_dict["environment"]["width"], 2),
        "height:",
        round(scenario_dict["environment"]["height"], 2),
    )
    print(
        "antennas:", len(scenario_dict["antennas"]),
        "humans:", len(scenario_dict["humans"]),
    )
    if floor_plan_summary is not None:
        print(
            "rooms:", floor_plan_summary["room_count"],
            "patios:", floor_plan_summary["patio_count"],
        )
    if target_summary is not None:
        print(
            "targets:", target_summary["target_count"],
            "valid_cells:", target_summary["valid_cell_count"],
        )
    if timing_summary is not None:
        print(
            "time_s:", round(timing_summary["total_seconds"], 2),
        )
    print()


def _setup_scenario_axis(ax: Axes, scenario_id: str, scenario: NetworkScenario, *, title_suffix: Optional[str] = None) -> Dict[str, Any]:
    scenario_dict = scenario.to_dict()
    xmin, xmax = scenario_dict["environment"]["x_domain"]
    ymin, ymax = scenario_dict["environment"]["y_range"]
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    title = scenario_id if title_suffix is None else f"{scenario_id} {title_suffix}"
    ax.set_title(title)
    return scenario_dict


def _scenario_figure_size(scenario_dict: Dict[str, Any]) -> tuple[float, float]:
    env = scenario_dict["environment"]
    width = max(float(env["width"]), 1.0)
    height = max(float(env["height"]), 1.0)

    if width >= height:
        short_edge = max(PLOT_MIN_SHORT_EDGE_INCHES, PLOT_LONG_EDGE_INCHES * (height / width))
        return (PLOT_LONG_EDGE_INCHES, short_edge)

    short_edge = max(PLOT_MIN_SHORT_EDGE_INCHES, PLOT_LONG_EDGE_INCHES * (width / height))
    return (short_edge, PLOT_LONG_EDGE_INCHES)


def _overlay_figure_size(scenario_dict: Dict[str, Any]) -> tuple[float, float]:
    env = scenario_dict["environment"]
    width = max(float(env["width"]), 1.0)
    height = max(float(env["height"]), 1.0)
    long_dim = max(width, height)
    long_edge_pixels = int(
        max(
            PLOT_MIN_LONG_EDGE_PIXELS,
            min(PLOT_MAX_LONG_EDGE_PIXELS, math.ceil(long_dim * PLOT_TARGET_PIXELS_PER_METER)),
        )
    )

    if width >= height:
        short_edge_pixels = max(1, int(round(long_edge_pixels * (height / width))))
        return (long_edge_pixels / PLOT_DPI, short_edge_pixels / PLOT_DPI)

    short_edge_pixels = max(1, int(round(long_edge_pixels * (width / height))))
    return (short_edge_pixels / PLOT_DPI, long_edge_pixels / PLOT_DPI)


def _draw_humans(ax: Axes, humans: List[Dict[str, Any]]) -> None:
    plt = _get_pyplot()
    human_color = "tab:red"
    for human in humans:
        if human.get("radius") is not None:
            circle = plt.Circle(
                human["position_X_Y"],
                human["radius"],
                color=human_color,
                alpha=0.35,
                ec="k",
                label="human",
            )
            ax.add_artist(circle)
        else:
            p0 = human["position_X_Y"]
            p1 = human.get("position_X1_Y1")
            if p1:
                width = p1[0] - p0[0]
                height = p1[1] - p0[1]
                rect = plt.Rectangle(
                    p0,
                    width,
                    height,
                    color=human_color,
                    alpha=0.35,
                    ec="k",
                    label="human",
                )
                ax.add_patch(rect)


def _draw_antennas(ax: Axes, antennas: List[Dict[str, Any]]) -> None:
    plt = _get_pyplot()
    for idx, antenna in enumerate(antennas):
        x, y = antenna["position"]
        radius = antenna["coverage_radius"]
        cover = plt.Circle(
            (x, y),
            radius,
            color="tab:blue",
            alpha=0.07,
            ec="tab:blue",
            label="coverage" if idx == 0 else None,
        )
        ax.add_artist(cover)
        ax.scatter(
            x,
            y,
            color="tab:blue",
            edgecolors="k",
            s=50,
            marker="^",
            label="antenna" if idx == 0 else None,
            zorder=5,
        )


def _draw_floor_plan(ax: Axes, plan: GeneratedFloorPlan) -> None:
    from renovation.elements import Door, Polygon, Wall, Window

    for patio in plan.patios:
        Polygon(
            vertices=[
                (patio.rect.x_min, patio.rect.y_min),
                (patio.rect.x_max, patio.rect.y_min),
                (patio.rect.x_max, patio.rect.y_max),
                (patio.rect.x_min, patio.rect.y_max),
            ],
            line_width=plan.config.patio_line_width,
            color=plan.config.patio_fill_color,
        ).draw(ax)

    for element in plan.elements:
        if element.element_type == "wall":
            Wall(
                anchor_point=(element.x, element.y),
                length=element.length or 0.0,
                thickness=element.thickness or 0.0,
                orientation_angle=element.orientation_angle,
            ).draw(ax)
        elif element.element_type == "door":
            Door(
                anchor_point=(element.x, element.y),
                doorway_width=element.doorway_width or 0.0,
                door_width=element.door_width or 0.0,
                thickness=element.thickness or 0.0,
                orientation_angle=element.orientation_angle,
                to_the_right=bool(element.to_the_right),
            ).draw(ax)
        elif element.element_type == "window":
            Window(
                anchor_point=(element.x, element.y),
                length=element.length or 0.0,
                overall_thickness=element.overall_thickness or 0.0,
                single_line_thickness=element.single_line_thickness or 0.0,
                orientation_angle=element.orientation_angle,
            ).draw(ax)


def _finalize_legend(ax: Axes) -> None:
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    if uniq:
        ax.legend(uniq.values(), uniq.keys(), loc="upper right")


def plot_scenario(scenario_id: str, scenario: NetworkScenario, out_dir: Path) -> Optional[str]:
    """
    Render the environment scenario (antennas + humans) and save to PNG.
    Returns the relative path to the saved image, or None on failure.
    """
    try:
        plt = _get_pyplot()
        scenario_dict = scenario.to_dict()
        fig, ax = plt.subplots(figsize=_scenario_figure_size(scenario_dict))
        scenario_dict = _setup_scenario_axis(ax, scenario_id, scenario, title_suffix="environment")
        _draw_humans(ax, scenario_dict["humans"])
        _draw_antennas(ax, scenario_dict["antennas"])
        _finalize_legend(ax)

        plots_dir = out_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        png_path = plots_dir / f"{scenario_id}_environment.png"
        fig.tight_layout()
        fig.savefig(png_path, dpi=PLOT_DPI, bbox_inches="tight")
        plt.close(fig)
        return str(png_path.relative_to(out_dir))
    except Exception:
        plt.close("all")
        return None


def plot_overlay_scenario(
    scenario_id: str,
    scenario: NetworkScenario,
    floor_plan: GeneratedFloorPlan,
    out_dir: Path,
) -> Optional[str]:
    """
    Render floor plan and environment entities on the same axis so both use the same scale.
    """
    try:
        plt = _get_pyplot()
        scenario_dict = scenario.to_dict()
        fig, ax = plt.subplots(figsize=_overlay_figure_size(scenario_dict))
        scenario_dict = _setup_scenario_axis(ax, scenario_id, scenario, title_suffix="overlay")
        _draw_floor_plan(ax, floor_plan)
        _draw_humans(ax, scenario_dict["humans"])
        _draw_antennas(ax, scenario_dict["antennas"])
        _finalize_legend(ax)

        plots_dir = out_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        png_path = plots_dir / f"{scenario_id}_overlay.png"
        fig.tight_layout()
        fig.savefig(png_path, dpi=PLOT_DPI, bbox_inches="tight")
        plt.close(fig)
        return str(png_path.relative_to(out_dir))
    except Exception:
        plt.close("all")
        return None


def plot_target_grid_scenario(
    scenario_id: str,
    scenario: NetworkScenario,
    floor_plan: GeneratedFloorPlan,
    target_summary: Dict[str, Any],
    out_dir: Path,
) -> Optional[str]:
    """
    Render a dedicated target-grid diagnostic plot.
    """
    try:
        plt = _get_pyplot()
        scenario_dict = scenario.to_dict()
        fig, ax = plt.subplots(figsize=_overlay_figure_size(scenario_dict))
        _setup_scenario_axis(ax, scenario_id, scenario, title_suffix="target grid")
        _draw_floor_plan(ax, floor_plan)

        valid_cells = [row for row in target_summary["grid_cell_rows"] if row["is_valid"]]
        invalid_cells = [row for row in target_summary["grid_cell_rows"] if not row["is_valid"]]

        if invalid_cells:
            ax.scatter(
                [row["x_center"] for row in invalid_cells],
                [row["y_center"] for row in invalid_cells],
                color="tab:gray",
                s=4,
                alpha=0.12,
                marker="s",
                label="invalid cell",
                zorder=2,
            )

        if valid_cells:
            ax.scatter(
                [row["x_center"] for row in valid_cells],
                [row["y_center"] for row in valid_cells],
                color="tab:green",
                s=5,
                alpha=0.55,
                marker="s",
                label="valid cell",
                zorder=3,
            )

        _draw_antennas(ax, scenario_dict["antennas"])
        _finalize_legend(ax)

        plots_dir = out_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        png_path = plots_dir / f"{scenario_id}_target_grid.png"
        fig.tight_layout()
        fig.savefig(png_path, dpi=PLOT_DPI, bbox_inches="tight")
        plt.close(fig)
        return str(png_path.relative_to(out_dir))
    except Exception:
        plt.close("all")
        return None


def generate_floor_plan_artifacts(
    scenario_id: str,
    seed: Optional[int],
    scenario: NetworkScenario,
    out_dir: Path,
    *,
    render_png: bool,
) -> Dict[str, Any]:
    floor_plan_out_dir = out_dir / "floor_plans" / scenario_id
    plan = generate_floor_plan_from_environment(
        scenario.environment,
        output_dir=floor_plan_out_dir,
        random_seed=seed,
        artifact_stem=f"{scenario_id}_floorplan",
        render_png=render_png,
    )

    png_path = None
    if plan.png_paths:
        png_path = str(plan.png_paths[0].relative_to(out_dir))

    element_rows: List[Dict[str, Any]] = []
    for element in plan.elements:
        row = asdict(element)
        row["scenario_id"] = scenario_id
        row["seed"] = seed
        element_rows.append(row)

    return {
        "yaml_path": str(plan.yaml_path.relative_to(out_dir)),
        "png_path": png_path,
        "parquet_path": str(plan.parquet_path.relative_to(out_dir)),
        "room_count": len(plan.rooms),
        "patio_count": len(plan.patios),
        "element_count": len(plan.elements),
        "element_rows": element_rows,
        "plan": plan,
    }


def generate_target_artifacts(
    scenario_id: str,
    seed: Optional[int],
    scenario: NetworkScenario,
    floor_plan_summary: Dict[str, Any],
) -> Dict[str, Any]:
    generated_grid = scenario.populate_targets_from_floor_plan(floor_plan_summary["plan"])

    grid_cell_rows: List[Dict[str, Any]] = []
    for cell in generated_grid.cells:
        row = asdict(cell)
        row["scenario_id"] = scenario_id
        row["seed"] = seed
        grid_cell_rows.append(row)

    target_rows: List[Dict[str, Any]] = []
    for target in generated_grid.targets:
        row = asdict(target)
        row["scenario_id"] = scenario_id
        row["seed"] = seed
        target_rows.append(row)

    return {
        "rows": generated_grid.rows,
        "cols": generated_grid.cols,
        "requested_max_cell_size": generated_grid.requested_max_cell_size,
        "cell_width": generated_grid.cell_width,
        "cell_height": generated_grid.cell_height,
        "grid_cell_count": len(generated_grid.cells),
        "valid_cell_count": len(generated_grid.targets),
        "target_count": len(generated_grid.targets),
        "grid_cell_rows": grid_cell_rows,
        "target_rows": target_rows,
    }


def generate_link_artifacts(
    scenario_id: str,
    seed: Optional[int],
    scenario: NetworkScenario,
    floor_plan_summary: Dict[str, Any],
) -> Dict[str, Any]:
    link_rows = build_link_rows(
        antennas=scenario.antennas,
        targets=scenario.targets,
        floor_plan=floor_plan_summary["plan"],
        humans=scenario.humans,
        scenario_id=scenario_id,
    )
    for row in link_rows:
        row["seed"] = seed

    return {
        "link_count": len(link_rows),
        "link_rows": link_rows,
    }


def scenario_to_rows(
    scenario_id: str,
    seed: Optional[int],
    scenario: NetworkScenario,
    plot_path: Optional[str],
    floor_plan_summary: Optional[Dict[str, Any]] = None,
    target_summary: Optional[Dict[str, Any]] = None,
    link_summary: Optional[Dict[str, Any]] = None,
    overlay_plot_path: Optional[str] = None,
    target_plot_path: Optional[str] = None,
    timing_summary: Optional[Dict[str, float]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Flatten a scenario into row dictionaries for each table."""
    scenario_dict = scenario.to_dict()

    env = scenario_dict["environment"]
    env_summary_row = {
        "scenario_id": scenario_id,
        "seed": seed,
        "area": env["area"],
        "env_type": env["env_type"],
        "width": env["width"],
        "height": env["height"],
        "x_domain_min": env["x_domain"][0],
        "x_domain_max": env["x_domain"][1],
        "y_range_min": env["y_range"][0],
        "y_range_max": env["y_range"][1],
        "antenna_count": len(scenario_dict["antennas"]),
        "human_count": len(scenario_dict["humans"]),
        "plot_path": plot_path,
        "overlay_plot_path": overlay_plot_path,
        "target_plot_path": target_plot_path,
        "floor_plan_yaml_path": floor_plan_summary["yaml_path"] if floor_plan_summary else None,
        "floor_plan_png_path": floor_plan_summary["png_path"] if floor_plan_summary else None,
        "floor_plan_parquet_path": floor_plan_summary["parquet_path"] if floor_plan_summary else None,
        "floor_plan_room_count": floor_plan_summary["room_count"] if floor_plan_summary else None,
        "floor_plan_patio_count": floor_plan_summary["patio_count"] if floor_plan_summary else None,
        "floor_plan_element_count": floor_plan_summary["element_count"] if floor_plan_summary else None,
        "target_count": target_summary["target_count"] if target_summary else None,
        "grid_cell_count": target_summary["grid_cell_count"] if target_summary else None,
        "valid_grid_cell_count": target_summary["valid_cell_count"] if target_summary else None,
        "target_grid_rows": target_summary["rows"] if target_summary else None,
        "target_grid_cols": target_summary["cols"] if target_summary else None,
        "target_requested_max_cell_size": target_summary["requested_max_cell_size"] if target_summary else None,
        "target_cell_width": target_summary["cell_width"] if target_summary else None,
        "target_cell_height": target_summary["cell_height"] if target_summary else None,
        "link_count": link_summary["link_count"] if link_summary else None,
        "scenario_generation_seconds": timing_summary["scenario_generation_seconds"] if timing_summary else None,
        "environment_plot_seconds": timing_summary["environment_plot_seconds"] if timing_summary else None,
        "floor_plan_seconds": timing_summary["floor_plan_seconds"] if timing_summary else None,
        "overlay_plot_seconds": timing_summary["overlay_plot_seconds"] if timing_summary else None,
        "total_seconds": timing_summary["total_seconds"] if timing_summary else None,
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

    human_rows = []
    for human in scenario_dict["humans"]:
        p0 = human["position_X_Y"]
        p1 = human.get("position_X1_Y1")
        human_rows.append(
            {
                "scenario_id": scenario_id,
                "seed": seed,
                "human_id": human["human_id"],
                "human_label": human.get("human_label"),
                "x0": p0[0],
                "y0": p0[1],
                "x1": p1[0] if p1 else None,
                "y1": p1[1] if p1 else None,
                "radius": human.get("radius"),
                "length": human.get("length"),
                "width": human.get("width"),
                "area": human["area"],
                "room_id": human.get("room_id"),
                "room_type": human.get("room_type"),
            }
        )

    return {
        "env_summary": [env_summary_row],
        "antennas": antenna_rows,
        "humans": human_rows,
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
    parser.add_argument(
        "--floorplan-plot",
        action="store_true",
        help="Render and save the standalone floor-plan PNG. If omitted, the YAML and parquet are still generated.",
    )
    parser.add_argument(
        "--target-plot",
        action="store_true",
        help="Render and save a dedicated target-grid PNG for each scenario.",
    )
    args = parser.parse_args()

    base_seed = args.seed
    out_dir = Path(args.output_dir)
    if not args.no_save:
        out_dir.mkdir(parents=True, exist_ok=True)

    # Collect rows for each table across all scenarios
    env_summary_rows: List[Dict[str, Any]] = []
    antenna_rows: List[Dict[str, Any]] = []
    human_rows: List[Dict[str, Any]] = []
    target_rows: List[Dict[str, Any]] = []
    grid_cell_rows: List[Dict[str, Any]] = []
    link_rows: List[Dict[str, Any]] = []
    floor_plan_element_rows: List[Dict[str, Any]] = []

    for iteration in range(1, args.count + 1):
        scenario_start = time.perf_counter()
        # Vary the seed per scenario so each run is different even with a base seed
        scenario_seed = None if base_seed is None else base_seed + (iteration - 1)
        scenario_id = f"scenario_{iteration:04d}"

        generation_start = time.perf_counter()
        network_scenario = NetworkScenario.generate_random(seed=scenario_seed)
        scenario_generation_seconds = time.perf_counter() - generation_start

        if not args.no_save:
            plot_path = None
            overlay_plot_path = None
            target_plot_path = None
            environment_plot_seconds = 0.0
            floor_plan_seconds = 0.0
            overlay_plot_seconds = 0.0

            floor_plan_start = time.perf_counter()
            floor_plan_summary = generate_floor_plan_artifacts(
                scenario_id,
                scenario_seed,
                network_scenario,
                out_dir,
                render_png=args.floorplan_plot,
            )
            floor_plan_seconds = time.perf_counter() - floor_plan_start
            network_scenario.populate_humans_from_floor_plan(
                floor_plan_summary["plan"],
                seed=scenario_seed,
            )
            target_summary = generate_target_artifacts(
                scenario_id,
                scenario_seed,
                network_scenario,
                floor_plan_summary,
            )
            link_summary = generate_link_artifacts(
                scenario_id,
                scenario_seed,
                network_scenario,
                floor_plan_summary,
            )
            if args.plot:
                plot_start = time.perf_counter()
                plot_path = plot_scenario(scenario_id, network_scenario, out_dir)
                environment_plot_seconds = time.perf_counter() - plot_start
            if args.plot:
                overlay_start = time.perf_counter()
                overlay_plot_path = plot_overlay_scenario(
                    scenario_id,
                    network_scenario,
                    floor_plan_summary["plan"],
                    out_dir,
                )
                overlay_plot_seconds = time.perf_counter() - overlay_start
            if args.target_plot:
                target_plot_path = plot_target_grid_scenario(
                    scenario_id,
                    network_scenario,
                    floor_plan_summary["plan"],
                    target_summary,
                    out_dir,
                )

            timing_summary = {
                "scenario_generation_seconds": scenario_generation_seconds,
                "environment_plot_seconds": environment_plot_seconds,
                "floor_plan_seconds": floor_plan_seconds,
                "overlay_plot_seconds": overlay_plot_seconds,
                "total_seconds": time.perf_counter() - scenario_start,
            }

            rows = scenario_to_rows(
                scenario_id,
                scenario_seed,
                network_scenario,
                plot_path,
                floor_plan_summary=floor_plan_summary,
                target_summary=target_summary,
                link_summary=link_summary,
                overlay_plot_path=overlay_plot_path,
                target_plot_path=target_plot_path,
                timing_summary=timing_summary,
            )
            summarize_scenario(
                scenario_id,
                network_scenario,
                floor_plan_summary=floor_plan_summary,
                target_summary=target_summary,
                timing_summary=timing_summary,
            )
            env_summary_rows.extend(rows["env_summary"])
            antenna_rows.extend(rows["antennas"])
            human_rows.extend(rows["humans"])
            target_rows.extend(target_summary["target_rows"])
            grid_cell_rows.extend(target_summary["grid_cell_rows"])
            link_rows.extend(link_summary["link_rows"])
            floor_plan_element_rows.extend(floor_plan_summary["element_rows"])
        else:
            floor_plan_start = time.perf_counter()
            with tempfile.TemporaryDirectory(prefix=f"{scenario_id}_floorplan_") as temp_dir:
                temp_plan = generate_floor_plan_from_environment(
                    network_scenario.environment,
                    output_dir=Path(temp_dir),
                    random_seed=scenario_seed,
                    artifact_stem=f"{scenario_id}_floorplan",
                    render_png=False,
                )
                network_scenario.populate_humans_from_floor_plan(
                    temp_plan,
                    seed=scenario_seed,
                )
                target_grid = network_scenario.populate_targets_from_floor_plan(temp_plan)
                floor_plan_summary = {
                    "room_count": len(temp_plan.rooms),
                    "patio_count": len(temp_plan.patios),
                }
                target_summary = {
                    "target_count": len(target_grid.targets),
                    "valid_cell_count": len(target_grid.targets),
                }
            floor_plan_seconds = time.perf_counter() - floor_plan_start
            timing_summary = {
                "scenario_generation_seconds": scenario_generation_seconds,
                "environment_plot_seconds": 0.0,
                "floor_plan_seconds": floor_plan_seconds,
                "overlay_plot_seconds": 0.0,
                "total_seconds": time.perf_counter() - scenario_start,
            }
            summarize_scenario(
                scenario_id,
                network_scenario,
                floor_plan_summary=floor_plan_summary,
                target_summary=target_summary,
                timing_summary=timing_summary,
            )

    if not args.no_save:
        pd.DataFrame(env_summary_rows).to_parquet(out_dir / "env_summary.parquet", index=False)
        pd.DataFrame(antenna_rows).to_parquet(out_dir / "antennas.parquet", index=False)
        pd.DataFrame(human_rows).to_parquet(out_dir / "humans.parquet", index=False)
        pd.DataFrame(target_rows).to_parquet(out_dir / "targets.parquet", index=False)
        pd.DataFrame(grid_cell_rows).to_parquet(out_dir / "grid_cells.parquet", index=False)
        pd.DataFrame(link_rows).to_parquet(out_dir / "links.parquet", index=False)
        pd.DataFrame(floor_plan_element_rows).to_parquet(out_dir / "floor_plan_elements.parquet", index=False)


if __name__ == "__main__":
    main()
