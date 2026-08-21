#!/usr/bin/env python3
"""Run the fixed-evaluation CartPole DAgger study for training seeds 0--4."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_STUDY_DIR = SCRIPT_DIR / "outputs" / "multiseed_fixed_eval_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=DEFAULT_STUDY_DIR)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--expert-timesteps", type=int, default=50_000)
    parser.add_argument("--dagger-timesteps", type=int, default=8_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    study_dir = args.study_dir.resolve()
    runs_dir = study_dir / "runs"
    logs_dir = study_dir / "logs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    for position, seed in enumerate(args.seeds, start=1):
        run_name = f"seed_{seed}"
        result_path = runs_dir / run_name / "results.json"
        if result_path.exists():
            print(
                f"[{position}/{len(args.seeds)}] seed {seed}: existing result kept",
                flush=True,
            )
            continue

        command = [
            sys.executable,
            str(SCRIPT_DIR / "run_library_dagger.py"),
            "--seed",
            str(seed),
            "--expert-timesteps",
            str(args.expert_timesteps),
            "--dagger-timesteps",
            str(args.dagger_timesteps),
            "--eval-episodes",
            str(args.eval_episodes),
            "--expert-eval-seed",
            "10000",
            "--final-eval-seed",
            "30000",
            "--round-eval-seed",
            "40000",
            "--output-root",
            str(runs_dir),
            "--run-name",
            run_name,
        ]
        log_path = logs_dir / f"seed_{seed}.log"
        print(
            f"[{position}/{len(args.seeds)}] seed {seed}: training started",
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
            f"[{position}/{len(args.seeds)}] seed {seed}: complete -> {result_path}",
            flush=True,
        )

    print(f"All requested seeds are complete in {study_dir}")


if __name__ == "__main__":
    main()
