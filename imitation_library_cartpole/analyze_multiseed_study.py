#!/usr/bin/env python3
"""Aggregate five-seed DAgger results and create publication-ready plots."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_STUDY_DIR = SCRIPT_DIR / "outputs" / "multiseed_fixed_eval_v1"
RETURN_CEILING = 500.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=DEFAULT_STUDY_DIR)
    return parser.parse_args()


def load_results(study_dir: Path) -> list[dict]:
    paths = sorted(
        (study_dir / "runs").glob("seed_*/results.json"),
        key=lambda path: int(path.parent.name.removeprefix("seed_")),
    )
    if not paths:
        raise SystemExit(f"No results.json files found under {study_dir / 'runs'}")
    results = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    seeds = [result["seed"] for result in results]
    if len(seeds) != len(set(seeds)):
        raise SystemExit(f"Duplicate training seeds found: {seeds}")
    return results


def confidence_summary(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    count = len(array)
    sample_std = float(np.std(array, ddof=1)) if count > 1 else 0.0
    standard_error = sample_std / np.sqrt(count)
    critical = float(student_t.ppf(0.975, count - 1)) if count > 1 else 0.0
    half_width = critical * standard_error
    mean = float(np.mean(array))
    return {
        "n_training_seeds": count,
        "mean": mean,
        "sample_standard_deviation": sample_std,
        "standard_error": float(standard_error),
        "t_critical_95_percent": critical,
        "ci95_half_width": float(half_width),
        "ci95_lower_unbounded": mean - half_width,
        "ci95_upper_unbounded": mean + half_width,
        "ci95_lower_bounded_for_cartpole": max(0.0, mean - half_width),
        "ci95_upper_bounded_for_cartpole": min(
            RETURN_CEILING, mean + half_width
        ),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    study_dir = args.study_dir.resolve()
    analysis_dir = study_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    results = load_results(study_dir)

    expected_round_count = len(results[0]["dagger"]["round_log"])
    for result in results:
        round_log = result["dagger"]["round_log"]
        if len(round_log) != expected_round_count:
            raise SystemExit("Runs do not have the same number of DAgger rounds")
        if any(entry["evaluation_seed"] != 40_000 for entry in round_log):
            raise SystemExit("A run does not use fixed round evaluation seed 40000")

    per_seed_round_rows = []
    for result in results:
        for entry in result["dagger"]["round_log"]:
            per_seed_round_rows.append(
                {
                    "training_seed": result["seed"],
                    "round": entry["round"],
                    "beta": entry["beta"],
                    "round_timesteps": entry["round_timesteps"],
                    "cumulative_dagger_timesteps": entry[
                        "cumulative_dagger_timesteps"
                    ],
                    "evaluation_seed": entry["evaluation_seed"],
                    "evaluation_episodes": entry["evaluation_episodes"],
                    "learner_mean_return": entry["learner_mean_return"],
                    "within_seed_episode_std": entry["learner_return_std"],
                }
            )

    round_aggregates = []
    for round_index in range(expected_round_count):
        entries = [
            result["dagger"]["round_log"][round_index] for result in results
        ]
        values = [entry["learner_mean_return"] for entry in entries]
        summary = confidence_summary(values)
        round_aggregates.append(
            {
                "round": round_index,
                "beta": entries[0]["beta"],
                "mean_cumulative_dagger_timesteps": float(
                    np.mean(
                        [entry["cumulative_dagger_timesteps"] for entry in entries]
                    )
                ),
                "mean_return": summary["mean"],
                "across_seed_sample_std": summary[
                    "sample_standard_deviation"
                ],
                "standard_error": summary["standard_error"],
                "ci95_half_width": summary["ci95_half_width"],
                "ci95_lower_unbounded": summary["ci95_lower_unbounded"],
                "ci95_upper_unbounded": summary["ci95_upper_unbounded"],
                "ci95_lower_bounded_for_plot": summary[
                    "ci95_lower_bounded_for_cartpole"
                ],
                "ci95_upper_bounded_for_plot": summary[
                    "ci95_upper_bounded_for_cartpole"
                ],
                "minimum_seed_mean": summary["minimum"],
                "maximum_seed_mean": summary["maximum"],
                "perfect_seed_count": sum(value == RETURN_CEILING for value in values),
            }
        )

    final_per_seed = []
    for result in results:
        final_per_seed.append(
            {
                "training_seed": result["seed"],
                "ppo_expert_mean_return": result["expert"]["mean_reward"],
                "ppo_expert_within_seed_episode_std": result["expert"][
                    "reward_standard_deviation"
                ],
                "final_dagger_mean_return": result["dagger"]["mean_reward"],
                "final_dagger_within_seed_episode_std": result["dagger"][
                    "reward_standard_deviation"
                ],
                "ppo_actual_training_timesteps": result["expert"][
                    "actual_training_timesteps"
                ],
                "dagger_actual_demonstration_timesteps": result["dagger"][
                    "actual_demonstration_timesteps"
                ],
            }
        )

    expert_summary = confidence_summary(
        [row["ppo_expert_mean_return"] for row in final_per_seed]
    )
    final_dagger_summary = confidence_summary(
        [row["final_dagger_mean_return"] for row in final_per_seed]
    )
    aggregate = {
        "study": "CartPole-v1 library DAgger fixed-evaluation five-seed baseline",
        "training_seeds": [result["seed"] for result in results],
        "evaluation": {
            "episodes_per_checkpoint": results[0]["evaluation_episodes"],
            "expert_seed": results[0]["derived_seeds"]["ppo_evaluation"],
            "round_seed": results[0]["derived_seeds"][
                "dagger_round_evaluation"
            ],
            "final_seed": results[0]["derived_seeds"]["dagger_evaluation"],
            "deterministic_actions": True,
        },
        "expert_final_across_training_seeds": expert_summary,
        "dagger_final_across_training_seeds": final_dagger_summary,
        "round_aggregates": round_aggregates,
        "software_versions": results[0]["versions"],
        "confidence_interval_definition": (
            "Two-sided 95% Student-t interval over independent training-seed "
            "means: mean +/- t_(0.975,n-1) * sample_std/sqrt(n). Plot bounds "
            "are clipped to CartPole's physical return range [0,500]."
        ),
    }

    write_csv(analysis_dir / "per_seed_round_results.csv", per_seed_round_rows)
    write_csv(analysis_dir / "round_aggregate.csv", round_aggregates)
    write_csv(analysis_dir / "final_per_seed.csv", final_per_seed)
    (analysis_dir / "aggregate_results.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    rounds = np.asarray([row["round"] for row in round_aggregates])
    means = np.asarray([row["mean_return"] for row in round_aggregates])
    lower = np.asarray(
        [row["ci95_lower_bounded_for_plot"] for row in round_aggregates]
    )
    upper = np.asarray(
        [row["ci95_upper_bounded_for_plot"] for row in round_aggregates]
    )

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axis = plt.subplots(figsize=(7.4, 4.3), constrained_layout=True)
    colors = plt.cm.Blues(np.linspace(0.38, 0.78, len(results)))
    for color, result in zip(colors, results):
        seed_values = [
            entry["learner_mean_return"]
            for entry in result["dagger"]["round_log"]
        ]
        axis.plot(
            rounds,
            seed_values,
            marker="o",
            linewidth=1.2,
            alpha=0.7,
            color=color,
            label=f"seed {result['seed']}",
        )
    axis.fill_between(
        rounds,
        lower,
        upper,
        color="#ef8354",
        alpha=0.22,
        label="95% t-CI across seeds",
    )
    axis.plot(
        rounds,
        means,
        marker="D",
        linewidth=2.5,
        color="#c44900",
        label="five-seed mean",
        zorder=5,
    )
    axis.axhline(
        RETURN_CEILING,
        color="#3d405b",
        linestyle="--",
        linewidth=1.0,
        label="return ceiling (500)",
    )
    axis.set_xlabel("DAgger round (checkpoint evaluated after retraining)")
    axis.set_ylabel("Mean return over 20 fixed episodes")
    axis.set_xticks(rounds)
    axis.set_ylim(0, 525)
    axis.grid(axis="y", alpha=0.22)
    axis.legend(ncol=2, loc="lower right", fontsize=8.4, frameon=False)
    axis.set_title("CartPole-v1: independent training seeds with common evaluation episodes")
    figure.savefig(analysis_dir / "multiseed_learning_curve.pdf")
    figure.savefig(analysis_dir / "multiseed_learning_curve.png", dpi=300)
    plt.close(figure)

    print(json.dumps(aggregate, indent=2, sort_keys=True))
    print(f"Analysis artifacts: {analysis_dir}")


if __name__ == "__main__":
    main()
