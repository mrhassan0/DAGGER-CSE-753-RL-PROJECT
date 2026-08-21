#!/usr/bin/env python3
"""Print/export human-readable summaries of DAgger demo trajectories.

Each `dagger/demos/round-XXX/*.npz` path saved by imitation's
SimpleDAggerTrainer is actually a HuggingFace `datasets` dataset directory,
not a real .npz. This script reads the underlying Arrow file directly with
pyarrow (no torch/gymnasium/imitation required) and prints one row per
trajectory: which round it came from, how many timesteps it lasted, the
total reward, and the left/right action split.

Usage:
    python3 inspect_dagger_demos.py [run_directory] [--csv out.csv]

If run_directory is omitted, every directory under outputs/ is scanned.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc

SCRIPT_DIR = Path(__file__).resolve().parent


def read_trajectories(npz_dir: Path) -> list[dict]:
    arrow_path = npz_dir / "data-00000-of-00001.arrow"
    with pa.memory_map(str(arrow_path), "r") as source:
        table = ipc.open_stream(source).read_all()
    data = table.to_pydict()
    rows = []
    for acts, rews, terminal in zip(data["acts"], data["rews"], data["terminal"]):
        rows.append(
            {
                "timesteps": len(acts),
                "total_reward": sum(rews),
                "terminal": terminal,
                "push_left": acts.count(0),
                "push_right": acts.count(1),
            }
        )
    return rows


def find_rounds(run_directory: Path) -> list[Path]:
    demos_dir = run_directory / "dagger" / "demos"
    if not demos_dir.is_dir():
        return []
    return sorted(demos_dir.glob("round-*"))


def summarize_run(run_directory: Path) -> list[dict]:
    summary_rows = []
    for round_dir in find_rounds(run_directory):
        npz_dirs = sorted(p for p in round_dir.glob("*.npz") if p.is_dir())
        for npz_dir in npz_dirs:
            for i, traj in enumerate(read_trajectories(npz_dir)):
                summary_rows.append(
                    {
                        "run": run_directory.name,
                        "round": round_dir.name,
                        "demo_file": npz_dir.name,
                        "trajectory_index": i,
                        **traj,
                    }
                )
    return summary_rows


def print_table(rows: list[dict]) -> None:
    if not rows:
        print("No demo trajectories found.")
        return

    columns = [
        "run",
        "round",
        "trajectory_index",
        "timesteps",
        "total_reward",
        "push_left",
        "push_right",
        "terminal",
        "demo_file",
    ]
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in columns}
    header = "  ".join(c.ljust(widths[c]) for c in columns)
    print(header)
    print("-" * len(header))
    for r in rows:
        print("  ".join(str(r[c]).ljust(widths[c]) for c in columns))

    # Per-round rollup
    print()
    print("Per-round summary:")
    rounds = sorted(set(r["round"] for r in rows))
    for rnd in rounds:
        round_rows = [r for r in rows if r["round"] == rnd]
        n = len(round_rows)
        mean_reward = sum(r["total_reward"] for r in round_rows) / n
        mean_len = sum(r["timesteps"] for r in round_rows) / n
        print(
            f"  {rnd}: {n} trajectories, "
            f"mean length {mean_len:.1f} steps, "
            f"mean reward {mean_reward:.1f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_directory",
        nargs="?",
        type=Path,
        help="A specific outputs/<run-name> directory (default: all runs under outputs/)",
    )
    parser.add_argument("--csv", type=Path, help="Optional path to write a CSV export")
    args = parser.parse_args()

    if args.run_directory:
        run_directories = [args.run_directory.resolve()]
    else:
        outputs_dir = SCRIPT_DIR / "outputs"
        run_directories = sorted(p for p in outputs_dir.iterdir() if p.is_dir())

    all_rows = []
    for run_directory in run_directories:
        all_rows.extend(summarize_run(run_directory))

    print_table(all_rows)

    if args.csv:
        with args.csv.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "run",
                    "round",
                    "demo_file",
                    "trajectory_index",
                    "timesteps",
                    "total_reward",
                    "push_left",
                    "push_right",
                    "terminal",
                ],
            )
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nWrote CSV: {args.csv}")


if __name__ == "__main__":
    main()
