#!/usr/bin/env python3
"""Run the original-paper-style comparison for training seeds 0--4."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_STUDY_DIR = SCRIPT_DIR / "outputs" / "paper_style_cartpole_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=DEFAULT_STUDY_DIR)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    study_directory = args.study_dir.resolve()
    runs_directory = study_directory / "runs"
    logs_directory = study_directory / "logs"
    runs_directory.mkdir(parents=True, exist_ok=True)
    logs_directory.mkdir(parents=True, exist_ok=True)

    for position, seed in enumerate(args.seeds, start=1):
        result_path = runs_directory / f"seed_{seed}" / "results.json"
        if result_path.exists():
            print(
                f"[{position}/{len(args.seeds)}] seed {seed}: existing result kept",
                flush=True,
            )
            continue
        command = [
            sys.executable,
            str(SCRIPT_DIR / "run_original_style_comparison.py"),
            "--seed",
            str(seed),
            "--output-root",
            str(runs_directory),
            "--run-name",
            f"seed_{seed}",
        ]
        log_path = logs_directory / f"seed_{seed}.log"
        print(
            f"[{position}/{len(args.seeds)}] seed {seed}: started",
            flush=True,
        )
        with log_path.open("w", encoding="utf-8") as log_file:
            completed = subprocess.run(
                command,
                cwd=SCRIPT_DIR,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0:
            raise SystemExit(
                f"Seed {seed} failed (exit {completed.returncode}); see {log_path}"
            )
        print(
            f"[{position}/{len(args.seeds)}] seed {seed}: complete",
            flush=True,
        )
    print(f"All requested seeds are complete in {study_directory}")


if __name__ == "__main__":
    main()
