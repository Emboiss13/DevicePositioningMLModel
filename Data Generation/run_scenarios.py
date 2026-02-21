"""
Generate one or more random network scenarios and print a short summary for each.

Usage examples
--------------
python3 run_scenarios.py --count 5
python3 run_scenarios.py --count 10 --seed 2
"""

import argparse

from generalStructure import NetworkScenario


def summarize_scenario(idx: int, scenario: NetworkScenario) -> None:
    dict = scenario.to_dict()
    print(f"Scenario {idx}")
    print(
        "  env:",
        dict["environment"]["env_type"],
        "area:",
        round(dict["environment"]["grid_area"], 2),
    )
    print(
        "  devices:", len(dict["devices"]),
        "obstacles:", len(dict["obstacles"]),
        "channels:", len(dict["channels"]),
    )
    print("  target_selected:", dict["target_selected"])
    print()


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
    args = parser.parse_args()

    base_seed = args.seed
    for i in range(1, args.count + 1):
        # Vary the seed per scenario so each run is different even with a base seed
        scenario_seed = None if base_seed is None else base_seed + (i - 1)
        s = NetworkScenario.generate_random(seed=scenario_seed)
        summarize_scenario(i, s)


if __name__ == "__main__":
    main()
