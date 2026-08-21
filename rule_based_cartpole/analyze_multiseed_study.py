#!/usr/bin/env python3
"""Aggregate rule-expert DAgger runs and compare them with the PPO baseline."""

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
DEFAULT_STUDY_DIR = SCRIPT_DIR / "outputs" / "rule_multiseed_fixed_eval_v1"
DEFAULT_PPO_STUDY_DIR = (
    SCRIPT_DIR.parent
    / "imitation_library_cartpole"
    / "outputs"
    / "multiseed_fixed_eval_v1"
)
RETURN_CEILING = 500.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=DEFAULT_STUDY_DIR)
    parser.add_argument("--ppo-study-dir", type=Path, default=DEFAULT_PPO_STUDY_DIR)
    return parser.parse_args()


def load_runs(study_dir: Path) -> list[dict]:
    paths = sorted(
        (study_dir / "runs").glob("seed_*/results.json"),
        key=lambda path: int(path.parent.name.removeprefix("seed_")),
    )
    if not paths:
        raise SystemExit(f"No results found under {study_dir / 'runs'}")
    results = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    seeds = [result["seed"] for result in results]
    if len(seeds) != len(set(seeds)):
        raise SystemExit(f"Duplicate training seeds found: {seeds}")
    return results


def confidence_summary(
    values: list[float],
    bounds: tuple[float, float] | None = (0.0, RETURN_CEILING),
) -> dict:
    array = np.asarray(values, dtype=np.float64)
    count = len(array)
    sample_std = float(np.std(array, ddof=1)) if count > 1 else 0.0
    standard_error = sample_std / np.sqrt(count)
    critical = float(student_t.ppf(0.975, count - 1)) if count > 1 else 0.0
    half_width = critical * standard_error
    mean = float(np.mean(array))
    bounded_lower = max(bounds[0], mean - half_width) if bounds else None
    bounded_upper = min(bounds[1], mean + half_width) if bounds else None
    return {
        "n_training_seeds": count,
        "mean": mean,
        "sample_standard_deviation": sample_std,
        "standard_error": float(standard_error),
        "t_critical_95_percent": critical,
        "ci95_half_width": float(half_width),
        "ci95_lower_unbounded": mean - half_width,
        "ci95_upper_unbounded": mean + half_width,
        "ci95_lower_bounded_for_cartpole": bounded_lower,
        "ci95_upper_bounded_for_cartpole": bounded_upper,
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


def aggregate_rounds(results: list[dict]) -> list[dict]:
    round_count = len(results[0]["dagger"]["round_log"])
    for result in results:
        if len(result["dagger"]["round_log"]) != round_count:
            raise SystemExit("Rule-expert runs do not have equal round counts")
        if any(
            entry["evaluation_seed"] != 40_000
            for entry in result["dagger"]["round_log"]
        ):
            raise SystemExit("A rule-expert run does not use evaluation seed 40000")

    aggregates = []
    for round_index in range(round_count):
        entries = [result["dagger"]["round_log"][round_index] for result in results]
        values = [entry["learner_mean_return"] for entry in entries]
        summary = confidence_summary(values)
        disagreements = [
            entry["learner_expert_disagreement_rate_before_update"]
            for entry in entries
            if entry["learner_expert_disagreement_rate_before_update"] is not None
        ]
        aggregates.append(
            {
                "round": round_index,
                "beta": entries[0]["beta"],
                "mean_round_labels": float(
                    np.mean([entry["round_timesteps"] for entry in entries])
                ),
                "minimum_round_labels": int(
                    np.min([entry["round_timesteps"] for entry in entries])
                ),
                "maximum_round_labels": int(
                    np.max([entry["round_timesteps"] for entry in entries])
                ),
                "mean_cumulative_labels": float(
                    np.mean(
                        [entry["cumulative_dagger_labels"] for entry in entries]
                    )
                ),
                "mean_realized_expert_control_fraction": float(
                    np.mean(
                        [
                            entry["realized_expert_control_fraction"]
                            for entry in entries
                        ]
                    )
                ),
                "mean_preupdate_learner_expert_disagreement_rate": (
                    float(np.mean(disagreements)) if disagreements else None
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
    return aggregates


def make_rule_plot(
    results: list[dict],
    aggregates: list[dict],
    analysis_dir: Path,
) -> None:
    rounds = np.asarray([row["round"] for row in aggregates])
    means = np.asarray([row["mean_return"] for row in aggregates])
    lower = np.asarray([row["ci95_lower_bounded_for_plot"] for row in aggregates])
    upper = np.asarray([row["ci95_upper_bounded_for_plot"] for row in aggregates])
    figure, axis = plt.subplots(figsize=(7.4, 4.3), constrained_layout=True)
    colors = plt.cm.Greens(np.linspace(0.38, 0.78, len(results)))
    for color, result in zip(colors, results):
        values = [
            entry["learner_mean_return"]
            for entry in result["dagger"]["round_log"]
        ]
        axis.plot(
            rounds,
            values,
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
        color="#4f772d",
        alpha=0.18,
        label="95% t-CI across seeds",
    )
    axis.plot(
        rounds,
        means,
        marker="D",
        linewidth=2.5,
        color="#31572c",
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
    axis.set_title("CartPole-v1 rule expert: five independent learner seeds")
    figure.savefig(analysis_dir / "rule_dagger_learning_curve.pdf")
    figure.savefig(analysis_dir / "rule_dagger_learning_curve.png", dpi=300)
    plt.close(figure)


def make_comparison_plot(
    rule_aggregates: list[dict],
    ppo_aggregates: list[dict],
    analysis_dir: Path,
) -> None:
    rounds = np.asarray([row["round"] for row in rule_aggregates])
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(7.4, 6.2),
        gridspec_kw={"height_ratios": [2.0, 1.25]},
        constrained_layout=True,
    )
    for axis in axes:
        for rows, color, label in (
            (rule_aggregates, "#31572c", "rule expert"),
            (ppo_aggregates, "#c44900", "PPO expert"),
        ):
            means = np.asarray([row["mean_return"] for row in rows])
            lower = np.asarray(
                [row["ci95_lower_bounded_for_plot"] for row in rows]
            )
            upper = np.asarray(
                [row["ci95_upper_bounded_for_plot"] for row in rows]
            )
            axis.fill_between(rounds, lower, upper, color=color, alpha=0.14)
            axis.plot(
                rounds,
                means,
                marker="o",
                linewidth=2.4,
                color=color,
                label=f"{label}: mean and 95% t-CI",
            )
        axis.axhline(
            RETURN_CEILING,
            color="#3d405b",
            linestyle="--",
            linewidth=1.0,
            label="return ceiling (500)",
        )
        axis.grid(axis="y", alpha=0.22)
        axis.set_ylabel("Mean return")

    axes[0].set_xticks(rounds)
    axes[0].set_ylim(0, 525)
    axes[0].legend(loc="lower right", fontsize=8.8, frameon=False)
    axes[0].set_title("Matched five-seed protocol: rule expert versus PPO expert")
    axes[1].set_xlim(0.8, 2.2)
    axes[1].set_xticks([1, 2])
    axes[1].set_ylim(470, 503)
    axes[1].set_title("Zoom: convergence at rounds 1--2", fontsize=10)
    axes[1].set_xlabel("DAgger round")
    figure.savefig(analysis_dir / "rule_vs_ppo_comparison.pdf")
    figure.savefig(analysis_dir / "rule_vs_ppo_comparison.png", dpi=300)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    study_dir = args.study_dir.resolve()
    ppo_study_dir = args.ppo_study_dir.resolve()
    analysis_dir = study_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    rule_results = load_runs(study_dir)
    ppo_results = load_runs(ppo_study_dir)
    rule_seeds = [result["seed"] for result in rule_results]
    ppo_seeds = [result["seed"] for result in ppo_results]
    if rule_seeds != ppo_seeds:
        raise SystemExit(
            f"Rule seeds {rule_seeds} do not match PPO seeds {ppo_seeds}"
        )

    rule_aggregates = aggregate_rounds(rule_results)
    ppo_aggregate_path = ppo_study_dir / "analysis" / "aggregate_results.json"
    if not ppo_aggregate_path.exists():
        raise SystemExit(f"Missing PPO aggregate result: {ppo_aggregate_path}")
    ppo_aggregates = json.loads(
        ppo_aggregate_path.read_text(encoding="utf-8")
    )["round_aggregates"]
    if len(rule_aggregates) != len(ppo_aggregates):
        raise SystemExit("Rule and PPO studies have different round counts")

    per_seed_round_rows = []
    for result in rule_results:
        for entry in result["dagger"]["round_log"]:
            per_seed_round_rows.append(
                {
                    "training_seed": result["seed"],
                    "round": entry["round"],
                    "beta": entry["beta"],
                    "round_labels": entry["round_timesteps"],
                    "cumulative_labels": entry["cumulative_dagger_labels"],
                    "realized_expert_control_fraction": entry[
                        "realized_expert_control_fraction"
                    ],
                    "preupdate_learner_expert_disagreement_rate": entry[
                        "learner_expert_disagreement_rate_before_update"
                    ],
                    "learner_mean_return": entry["learner_mean_return"],
                    "within_seed_episode_std": entry["learner_return_std"],
                }
            )

    final_per_seed = []
    for result in rule_results:
        final_per_seed.append(
            {
                "training_seed": result["seed"],
                "rule_expert_mean_return": result["expert"]["mean_reward"],
                "final_rule_dagger_mean_return": result["dagger"]["mean_reward"],
                "final_rule_dagger_within_seed_episode_std": result["dagger"][
                    "reward_standard_deviation"
                ],
                "actual_total_labels": result["dagger"]["actual_total_labels"],
            }
        )

    comparison_rows = []
    for rule_row, ppo_row in zip(rule_aggregates, ppo_aggregates):
        comparison_rows.append(
            {
                "round": rule_row["round"],
                "rule_mean_return": rule_row["mean_return"],
                "rule_across_seed_sample_std": rule_row[
                    "across_seed_sample_std"
                ],
                "rule_ci95_lower": rule_row["ci95_lower_unbounded"],
                "rule_ci95_upper": rule_row["ci95_upper_unbounded"],
                "ppo_mean_return": ppo_row["mean_return"],
                "ppo_across_seed_sample_std": ppo_row[
                    "across_seed_sample_std"
                ],
                "ppo_ci95_lower": ppo_row["ci95_lower_unbounded"],
                "ppo_ci95_upper": ppo_row["ci95_upper_unbounded"],
                "rule_minus_ppo_mean_return": (
                    rule_row["mean_return"] - ppo_row["mean_return"]
                ),
            }
        )

    final_summary = confidence_summary(
        [row["final_rule_dagger_mean_return"] for row in final_per_seed]
    )
    label_summary = confidence_summary(
        [float(row["actual_total_labels"]) for row in final_per_seed],
        bounds=None,
    )
    aggregate = {
        "study": "CartPole-v1 rule-expert DAgger fixed-evaluation five-seed study",
        "training_seeds": rule_seeds,
        "expert_formula": "1[theta + 0.6*theta_dot + 0.02*x_dot > 0]",
        "initial_dataset_sha256": rule_results[0]["initial_dataset"]["sha256"],
        "initial_dataset_selected_transitions": rule_results[0]["initial_dataset"][
            "selected_transitions"
        ],
        "evaluation": rule_results[0]["evaluation_protocol"],
        "rule_expert_evaluation": {
            "mean_return": rule_results[0]["expert"]["mean_reward"],
            "episode_standard_deviation": rule_results[0]["expert"][
                "reward_standard_deviation"
            ],
        },
        "dagger_final_across_training_seeds": final_summary,
        "actual_total_labels_across_training_seeds": label_summary,
        "round_aggregates": rule_aggregates,
        "comparison_with_ppo": comparison_rows,
        "software_versions": rule_results[0]["versions"],
        "confidence_interval_definition": (
            "Two-sided 95% Student-t interval over independent learner-training "
            "seed means: mean +/- t_(0.975,n-1) * sample_std/sqrt(n)."
        ),
    }

    write_csv(analysis_dir / "per_seed_round_results.csv", per_seed_round_rows)
    write_csv(analysis_dir / "round_aggregate.csv", rule_aggregates)
    write_csv(analysis_dir / "final_per_seed.csv", final_per_seed)
    write_csv(analysis_dir / "comparison_with_ppo.csv", comparison_rows)
    (analysis_dir / "aggregate_results.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    make_rule_plot(rule_results, rule_aggregates, analysis_dir)
    make_comparison_plot(rule_aggregates, ppo_aggregates, analysis_dir)
    print(json.dumps(aggregate, indent=2, sort_keys=True))
    print(f"Analysis artifacts: {analysis_dir}")


if __name__ == "__main__":
    main()
