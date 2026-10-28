"""Launch manuscript experiment groups sequentially."""

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXPERIMENTS = (
    "reference_geometry",
    "ablation_50x100",
    "penetration_sweep",
    "large_seed_stability",
    "forward_data",
    "likelihood_cost",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--experiment", choices=EXPERIMENTS)
    group.add_argument("--all", action="store_true", help="Run every group; original IDCS scale sweep may take days")
    group.add_argument("--list", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Small code checks; not manuscript results")
    args = parser.parse_args()
    if args.list:
        print("\n".join(EXPERIMENTS))
        return
    selected = EXPERIMENTS if args.all else (args.experiment,)
    for name in selected:
        command = [sys.executable, str(ROOT / "experiments" / name / "run.py")]
        if args.smoke and name != "reference_geometry":
            command.append("--smoke")
        print(f"Running {name}", flush=True)
        subprocess.run(command, cwd=str(ROOT), check=True)


if __name__ == "__main__":
    main()
