#!/usr/bin/env python3
"""Aggregate five-seed results and make paper-style comparison figures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_STUDY_DIR = SCRIPT_DIR / "outputs" / "paper_style_cartpole_v1"
METHODS = ("dagger", "smile", "supervised")
DISPLAY_NAMES = {
    "dagger": r"DAgger ($\beta_i=\mathbf{1}[i=1]$)",
    "smile": r"SMILe ($\alpha=0.1$)",
    "supervised": "Supervised",
}
COLORS = {
    "dagger": "#1769aa",
    "smile": "#d1495b",
    "supervised": "#555555",
}
MARKERS = {"dagger": "o", "smile": "s", "supervised": "^"}
CHECKPOINTS = (1, 5, 10, 15, 20)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=DEFAULT_STUDY_DIR)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    return parser.parse_args()


def confidence_summary(values: list[float], lower: float, upper: float) -> dict:
    data = np.asarray(values, dtype=np.float64)
    count = len(data)
    mean = float(np.mean(data))
    sample_sd = float(np.std(data, ddof=1)) if count > 1 else 0.0
    standard_error = sample_sd / np.sqrt(count) if count > 1 else 0.0
    critical = float(t.ppf(0.975, df=count - 1)) if count > 1 else 0.0
    margin = critical * standard_error
    return {
        "n_seeds": count,
        "mean": mean,
        "sample_standard_deviation": sample_sd,
        "standard_error": standard_error,
        "t_critical_95_percent": critical,
        "confidence_interval_95_lower": max(lower, mean - margin),
        "confidence_interval_95_upper": min(upper, mean + margin),
        "unclipped_margin_of_error_95": margin,
        "seed_values": [float(value) for value in data],
    }


def load_results(study_directory: Path, seeds: list[int]) -> list[dict]:
    results = []
    for seed in seeds:
        result_path = study_directory / "runs" / f"seed_{seed}" / "results.json"
        if not result_path.exists():
            raise SystemExit(f"Missing result: {result_path}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result["seed"] != seed:
            raise SystemExit(f"Seed mismatch in {result_path}")
        results.append(result)
    return results


def validate_results(results: list[dict]) -> tuple[int, int]:
    reference_protocol = results[0]["protocol"]
    iterations = int(reference_protocol["iterations"])
    labels_per_iteration = int(reference_protocol["labels_per_iteration"])
    if iterations < max(CHECKPOINTS):
        raise SystemExit("Study has fewer than 20 iterations")

    for result in results:
        protocol = result["protocol"]
        checked_keys = (
            "iterations",
            "labels_per_iteration",
            "evaluation_episodes_per_checkpoint",
            "evaluation_seed",
            "bc_epochs_per_iteration",
            "smile_alpha",
        )
        for key in checked_keys:
            if protocol[key] != reference_protocol[key]:
                raise SystemExit(f"Protocol mismatch for {key}")
        if tuple(sorted(result["methods"])) != tuple(sorted(METHODS)):
            raise SystemExit("Method set mismatch")
        for method in METHODS:
            rows = result["methods"][method]
            if len(rows) != iterations:
                raise SystemExit(f"Wrong iteration count for {method}")
            for index, row in enumerate(rows, start=1):
                if row["iteration"] != index:
                    raise SystemExit(f"Iteration ordering mismatch for {method}")
                if row["new_labels"] != labels_per_iteration:
                    raise SystemExit(f"Unequal label budget for {method}")
                expected_total = index * labels_per_iteration
                if row["cumulative_labels"] != expected_total:
                    raise SystemExit(f"Cumulative-label mismatch for {method}")
                evaluation = row["evaluation"]
                if evaluation["failure_count"] != sum(
                    evaluation["episode_failures"]
                ):
                    raise SystemExit(f"Failure-count mismatch for {method}")
    return iterations, labels_per_iteration


def aggregate_results(results: list[dict], iterations: int) -> dict:
    aggregate: dict[str, list[dict]] = {method: [] for method in METHODS}
    for method in METHODS:
        for iteration_index in range(iterations):
            rows = [result["methods"][method][iteration_index] for result in results]
            returns = [row["evaluation"]["mean_return"] for row in rows]
            failures = [row["evaluation"]["failure_rate"] for row in rows]
            perfect_episode_counts = [
                sum(length == 500 for length in row["evaluation"]["episode_lengths"])
                for row in rows
            ]
            aggregate[method].append(
                {
                    "iteration": iteration_index + 1,
                    "new_labels_per_seed": rows[0]["new_labels"],
                    "cumulative_labels_per_seed": rows[0]["cumulative_labels"],
                    "failure_rate": confidence_summary(failures, 0.0, 1.0),
                    "mean_return": confidence_summary(returns, 0.0, 500.0),
                    "perfect_evaluation_episodes_across_seeds": int(
                        sum(perfect_episode_counts)
                    ),
                    "total_evaluation_episodes_across_seeds": int(
                        sum(row["evaluation"]["episodes"] for row in rows)
                    ),
                }
            )
    return aggregate


def write_per_seed_csv(
    results: list[dict], output_path: Path, iterations: int
) -> None:
    fields = [
        "seed",
        "method",
        "iteration",
        "new_labels",
        "cumulative_labels",
        "mean_return",
        "return_standard_deviation_across_episodes",
        "failure_rate",
        "failure_count",
        "evaluation_episodes",
        "training_failures",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for result in results:
            for method in METHODS:
                for index in range(iterations):
                    row = result["methods"][method][index]
                    evaluation = row["evaluation"]
                    writer.writerow(
                        {
                            "seed": result["seed"],
                            "method": method,
                            "iteration": row["iteration"],
                            "new_labels": row["new_labels"],
                            "cumulative_labels": row["cumulative_labels"],
                            "mean_return": evaluation["mean_return"],
                            "return_standard_deviation_across_episodes": (
                                evaluation["return_standard_deviation"]
                            ),
                            "failure_rate": evaluation["failure_rate"],
                            "failure_count": evaluation["failure_count"],
                            "evaluation_episodes": evaluation["episodes"],
                            "training_failures": row["training_failures"],
                        }
                    )


def write_aggregate_csv(aggregate: dict, output_path: Path) -> None:
    fields = [
        "method",
        "iteration",
        "cumulative_labels_per_seed",
        "metric",
        "n_seeds",
        "mean",
        "sample_standard_deviation",
        "standard_error",
        "confidence_interval_95_lower",
        "confidence_interval_95_upper",
        "unclipped_margin_of_error_95",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for method in METHODS:
            for row in aggregate[method]:
                for metric in ("failure_rate", "mean_return"):
                    summary = row[metric]
                    writer.writerow(
                        {
                            "method": method,
                            "iteration": row["iteration"],
                            "cumulative_labels_per_seed": row[
                                "cumulative_labels_per_seed"
                            ],
                            "metric": metric,
                            "n_seeds": summary["n_seeds"],
                            "mean": summary["mean"],
                            "sample_standard_deviation": summary[
                                "sample_standard_deviation"
                            ],
                            "standard_error": summary["standard_error"],
                            "confidence_interval_95_lower": summary[
                                "confidence_interval_95_lower"
                            ],
                            "confidence_interval_95_upper": summary[
                                "confidence_interval_95_upper"
                            ],
                            "unclipped_margin_of_error_95": summary[
                                "unclipped_margin_of_error_95"
                            ],
                        }
                    )


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "figure.dpi": 140,
            "savefig.dpi": 300,
        }
    )


def plot_metric(
    aggregate: dict,
    metric: str,
    ylabel: str,
    title: str,
    output_stem: Path,
    checkpoints: tuple[int, ...],
    ylim: tuple[float, float],
    reference_value: float,
    reference_label: str,
) -> None:
    figure, axis = plt.subplots(figsize=(5.15, 3.45), constrained_layout=True)
    for method in METHODS:
        rows = [aggregate[method][iteration - 1] for iteration in checkpoints]
        x_values = np.asarray(
            [row["cumulative_labels_per_seed"] for row in rows], dtype=float
        )
        means = np.asarray([row[metric]["mean"] for row in rows], dtype=float)
        lowers = np.asarray(
            [row[metric]["confidence_interval_95_lower"] for row in rows],
            dtype=float,
        )
        uppers = np.asarray(
            [row[metric]["confidence_interval_95_upper"] for row in rows],
            dtype=float,
        )
        asymmetric_errors = np.vstack((means - lowers, uppers - means))
        axis.errorbar(
            x_values,
            means,
            yerr=asymmetric_errors,
            color=COLORS[method],
            marker=MARKERS[method],
            markersize=4.5,
            linewidth=1.5,
            elinewidth=1.0,
            capsize=3,
            label=DISPLAY_NAMES[method],
        )

    axis.axhline(
        reference_value,
        color="#222222",
        linestyle="--",
        linewidth=0.9,
        alpha=0.7,
        label=reference_label,
    )
    axis.set_xlabel("Cumulative expert-labeled states per seed")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.set_ylim(*ylim)
    axis.legend(
        frameon=True,
        facecolor="white",
        edgecolor="#dddddd",
        framealpha=0.94,
        loc="center right",
    )
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".png"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    study_directory = args.study_dir.resolve()
    analysis_directory = study_directory / "analysis"
    analysis_directory.mkdir(parents=True, exist_ok=True)
    results = load_results(study_directory, args.seeds)
    iterations, labels_per_iteration = validate_results(results)
    aggregate = aggregate_results(results, iterations)

    payload = {
        "study_directory": str(study_directory),
        "training_seeds": args.seeds,
        "confidence_interval": (
            "two-sided 95% Student-t interval across per-seed means; bounded "
            "to the metric's feasible range only for plotting/reporting"
        ),
        "iterations": iterations,
        "labels_per_iteration_per_method_per_seed": labels_per_iteration,
        "evaluation_episodes_per_seed_per_checkpoint": results[0]["protocol"][
            "evaluation_episodes_per_checkpoint"
        ],
        "aggregate": aggregate,
    }
    (analysis_directory / "aggregate_results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_per_seed_csv(
        results,
        analysis_directory / "per_seed_results.csv",
        iterations,
    )
    write_aggregate_csv(
        aggregate,
        analysis_directory / "iteration_aggregate.csv",
    )

    set_plot_style()
    plot_metric(
        aggregate,
        metric="failure_rate",
        ylabel="Failures per evaluation episode (lower is better)",
        title="CartPole comparison at paper-matched checkpoints",
        output_stem=analysis_directory / "paper_style_failure_comparison",
        checkpoints=CHECKPOINTS,
        ylim=(-0.03, 1.03),
        reference_value=0.0,
        reference_label="Rule expert",
    )
    plot_metric(
        aggregate,
        metric="mean_return",
        ylabel="Mean return (maximum 500)",
        title="CartPole return at paper-matched checkpoints",
        output_stem=analysis_directory / "paper_style_return_comparison",
        checkpoints=CHECKPOINTS,
        ylim=(0.0, 515.0),
        reference_value=500.0,
        reference_label="Rule expert / ceiling",
    )
    plot_metric(
        aggregate,
        metric="failure_rate",
        ylabel="Failures per evaluation episode (lower is better)",
        title="CartPole comparison across all 20 iterations",
        output_stem=analysis_directory / "all_iterations_failure_comparison",
        checkpoints=tuple(range(1, iterations + 1)),
        ylim=(-0.03, 1.03),
        reference_value=0.0,
        reference_label="Rule expert",
    )
    plot_metric(
        aggregate,
        metric="mean_return",
        ylabel="Mean return (maximum 500)",
        title="CartPole return across all 20 iterations",
        output_stem=analysis_directory / "all_iterations_return_comparison",
        checkpoints=tuple(range(1, iterations + 1)),
        ylim=(0.0, 515.0),
        reference_value=500.0,
        reference_label="Rule expert / ceiling",
    )
    print(f"Analysis written to {analysis_directory}")


if __name__ == "__main__":
    main()
