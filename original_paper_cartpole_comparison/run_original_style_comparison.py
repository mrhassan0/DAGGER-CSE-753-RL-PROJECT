#!/usr/bin/env python3
"""Compare DAgger, SMILe, and supervised learning on rule-expert CartPole."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import gymnasium as gym
import imitation
import numpy as np
import stable_baselines3
import torch
from imitation.algorithms import bc
from imitation.data import types


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = (
    SCRIPT_DIR / "outputs" / "paper_style_cartpole_v1" / "runs"
)
ActionSelector = Callable[[np.ndarray, np.random.Generator], tuple[int, int]]


def rule_expert(observation: np.ndarray) -> int:
    _, cart_velocity, pole_angle, pole_angular_velocity = observation
    score = pole_angle + 0.6 * pole_angular_velocity + 0.02 * cart_velocity
    return int(score > 0.0)


def make_spaces():
    environment = gym.make("CartPole-v1")
    observation_space = environment.observation_space
    action_space = environment.action_space
    environment.close()
    return observation_space, action_space


def make_bc_trainer(seed: int) -> bc.BC:
    torch.manual_seed(seed)
    observation_space, action_space = make_spaces()
    return bc.BC(
        observation_space=observation_space,
        action_space=action_space,
        rng=np.random.default_rng(seed),
        device="cpu",
    )


def make_transitions(
    observations: np.ndarray,
    expert_actions: np.ndarray,
) -> types.Transitions:
    observations = np.asarray(observations, dtype=np.float32)
    infos = np.empty(len(observations), dtype=object)
    infos[:] = [{} for _ in range(len(observations))]
    return types.Transitions(
        obs=observations,
        acts=np.asarray(expert_actions, dtype=np.int64),
        infos=infos,
        next_obs=observations.copy(),
        dones=np.zeros(len(observations), dtype=bool),
    )


def fit_bc(
    trainer: bc.BC,
    observations: np.ndarray,
    expert_actions: np.ndarray,
    epochs: int,
) -> None:
    trainer.set_demonstrations(make_transitions(observations, expert_actions))
    trainer.train(n_epochs=epochs, progress_bar=False)


def policy_action(policy, observation: np.ndarray) -> int:
    action, _ = policy.predict(observation, deterministic=True)
    return int(np.asarray(action).item())


def collect_training_block(
    selector: ActionSelector,
    environment_seed: int,
    selector_seed: int,
    transitions: int,
) -> dict[str, np.ndarray]:
    """Collect an exact-size block, resetting CartPole after each failure."""

    environment = gym.make("CartPole-v1")
    environment.action_space.seed(environment_seed)
    observation, _ = environment.reset(seed=environment_seed)
    rng = np.random.default_rng(selector_seed)

    observations = []
    expert_actions = []
    executed_actions = []
    controller_ids = []
    rewards = []
    terminated_flags = []
    truncated_flags = []
    segment_ids = []
    timesteps = []
    segment_id = 0
    timestep = 0

    while len(observations) < transitions:
        observation_array = np.asarray(observation, dtype=np.float32)
        expert_action = rule_expert(observation_array)
        executed_action, controller_id = selector(observation_array, rng)
        next_observation, reward, terminated, truncated, _ = environment.step(
            executed_action
        )

        observations.append(observation_array)
        expert_actions.append(expert_action)
        executed_actions.append(executed_action)
        controller_ids.append(controller_id)
        rewards.append(float(reward))
        terminated_flags.append(bool(terminated))
        truncated_flags.append(bool(truncated))
        segment_ids.append(segment_id)
        timesteps.append(timestep)

        observation = next_observation
        timestep += 1
        if (terminated or truncated) and len(observations) < transitions:
            segment_id += 1
            timestep = 0
            observation, _ = environment.reset()

    environment.close()
    return {
        "observations": np.asarray(observations, dtype=np.float32),
        "expert_actions": np.asarray(expert_actions, dtype=np.int64),
        "executed_actions": np.asarray(executed_actions, dtype=np.int64),
        "controller_ids": np.asarray(controller_ids, dtype=np.int64),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "terminated": np.asarray(terminated_flags, dtype=bool),
        "truncated": np.asarray(truncated_flags, dtype=bool),
        "segment_ids": np.asarray(segment_ids, dtype=np.int64),
        "timesteps": np.asarray(timesteps, dtype=np.int64),
    }


def evaluate_selector(
    selector: ActionSelector,
    environment_seed: int,
    selector_seed: int,
    episodes: int,
) -> dict:
    """Evaluate on the common fixed episode sequence without expert help."""

    environment = gym.make("CartPole-v1")
    environment.action_space.seed(environment_seed)
    environment.reset(seed=environment_seed)
    # Match the existing project's fixed-evaluation reset sequence.
    observation, _ = environment.reset()
    episode_returns = []
    episode_lengths = []
    episode_failures = []

    for episode_index in range(episodes):
        rng = np.random.default_rng(selector_seed + episode_index)
        episode_return = 0.0
        episode_length = 0
        failed = False
        while True:
            action, controller_id = selector(
                np.asarray(observation, dtype=np.float32),
                rng,
            )
            if controller_id == -1:
                raise RuntimeError("Expert control is forbidden during evaluation")
            observation, reward, terminated, truncated, _ = environment.step(action)
            episode_return += float(reward)
            episode_length += 1
            if terminated or truncated:
                failed = bool(terminated)
                break
        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)
        episode_failures.append(int(failed))
        if episode_index + 1 < episodes:
            observation, _ = environment.reset()
    environment.close()

    returns = np.asarray(episode_returns, dtype=np.float64)
    failures = np.asarray(episode_failures, dtype=np.float64)
    return {
        "environment_seed": environment_seed,
        "selector_seed": selector_seed,
        "episodes": episodes,
        "mean_return": float(np.mean(returns)),
        "return_standard_deviation": float(np.std(returns, ddof=0)),
        "failure_rate": float(np.mean(failures)),
        "failure_count": int(np.sum(failures)),
        "episode_returns": episode_returns,
        "episode_lengths": episode_lengths,
        "episode_failures": episode_failures,
    }


def expert_selector(
    observation: np.ndarray,
    rng: np.random.Generator,
) -> tuple[int, int]:
    del rng
    return rule_expert(observation), -1


def learner_selector(policy) -> ActionSelector:
    def select(
        observation: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[int, int]:
        del rng
        return policy_action(policy, observation), 0

    return select


def smile_collection_selector(
    base_policies: list,
    alpha: float,
) -> ActionSelector:
    """Return pi_(n-1), including its remaining expert mixture weight."""

    learned_weights = np.asarray(
        [alpha * (1.0 - alpha) ** index for index in range(len(base_policies))],
        dtype=np.float64,
    )
    expert_weight = (1.0 - alpha) ** len(base_policies)
    probabilities = np.concatenate(([expert_weight], learned_weights))
    probabilities /= np.sum(probabilities)

    def select(
        observation: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[int, int]:
        component = int(rng.choice(len(probabilities), p=probabilities))
        if component == 0:
            return rule_expert(observation), -1
        policy_index = component - 1
        return policy_action(base_policies[policy_index], observation), policy_index + 1

    return select


def smile_expert_free_selector(
    base_policies: list,
    alpha: float,
) -> ActionSelector:
    """Return the normalized learned-policy mixture with no expert component."""

    if not base_policies:
        raise ValueError("SMILe evaluation requires at least one learned policy")
    weights = np.asarray(
        [alpha * (1.0 - alpha) ** index for index in range(len(base_policies))],
        dtype=np.float64,
    )
    probabilities = weights / np.sum(weights)

    def select(
        observation: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[int, int]:
        policy_index = int(rng.choice(len(probabilities), p=probabilities))
        return policy_action(base_policies[policy_index], observation), policy_index + 1

    return select


def make_iteration_seeds(training_seed: int, iteration: int, method_index: int):
    environment_seed = 100_000 + training_seed * 1_000 + iteration
    selector_seed = 200_000 + training_seed * 10_000 + method_index * 100 + iteration
    return environment_seed, selector_seed


def run_supervised(args, run_directory: Path) -> list[dict]:
    method_directory = run_directory / "supervised"
    data_directory = method_directory / "data"
    checkpoint_directory = method_directory / "checkpoints"
    data_directory.mkdir(parents=True)
    checkpoint_directory.mkdir()
    trainer = make_bc_trainer(args.seed)
    observation_blocks = []
    action_blocks = []
    log = []

    for iteration in range(1, args.iterations + 1):
        environment_seed, selector_seed = make_iteration_seeds(
            args.seed, iteration, 0
        )
        block = collect_training_block(
            expert_selector,
            environment_seed,
            selector_seed,
            args.labels_per_iteration,
        )
        np.savez_compressed(
            data_directory / f"iteration_{iteration:03d}.npz", **block
        )
        observation_blocks.append(block["observations"])
        action_blocks.append(block["expert_actions"])
        aggregate_observations = np.concatenate(observation_blocks)
        aggregate_actions = np.concatenate(action_blocks)
        fit_bc(trainer, aggregate_observations, aggregate_actions, args.bc_epochs)
        torch.save(
            trainer.policy.state_dict(),
            checkpoint_directory / f"iteration_{iteration:03d}.pt",
        )
        evaluation = evaluate_selector(
            learner_selector(trainer.policy),
            args.evaluation_seed,
            300_000 + args.seed * 1_000,
            args.eval_episodes,
        )
        log.append(
            {
                "iteration": iteration,
                "new_labels": len(block["observations"]),
                "cumulative_labels": len(aggregate_observations),
                "training_failures": int(np.sum(block["terminated"])),
                "training_segments": int(np.max(block["segment_ids"]) + 1),
                "evaluation": evaluation,
            }
        )
    return log


def run_dagger(args, run_directory: Path) -> list[dict]:
    method_directory = run_directory / "dagger"
    data_directory = method_directory / "data"
    checkpoint_directory = method_directory / "checkpoints"
    data_directory.mkdir(parents=True)
    checkpoint_directory.mkdir()
    trainer = make_bc_trainer(args.seed)
    observation_blocks = []
    action_blocks = []
    log = []

    for iteration in range(1, args.iterations + 1):
        environment_seed, selector_seed = make_iteration_seeds(
            args.seed, iteration, 1
        )
        beta = 1.0 if iteration == 1 else 0.0
        selector = expert_selector if iteration == 1 else learner_selector(trainer.policy)
        block = collect_training_block(
            selector,
            environment_seed,
            selector_seed,
            args.labels_per_iteration,
        )
        np.savez_compressed(
            data_directory / f"iteration_{iteration:03d}.npz", **block
        )
        disagreement = None
        if iteration > 1:
            predicted_actions = np.asarray(
                [
                    policy_action(trainer.policy, observation)
                    for observation in block["observations"]
                ],
                dtype=np.int64,
            )
            disagreement = float(np.mean(predicted_actions != block["expert_actions"]))
        observation_blocks.append(block["observations"])
        action_blocks.append(block["expert_actions"])
        aggregate_observations = np.concatenate(observation_blocks)
        aggregate_actions = np.concatenate(action_blocks)
        fit_bc(trainer, aggregate_observations, aggregate_actions, args.bc_epochs)
        torch.save(
            trainer.policy.state_dict(),
            checkpoint_directory / f"iteration_{iteration:03d}.pt",
        )
        evaluation = evaluate_selector(
            learner_selector(trainer.policy),
            args.evaluation_seed,
            300_000 + args.seed * 1_000,
            args.eval_episodes,
        )
        log.append(
            {
                "iteration": iteration,
                "beta": beta,
                "new_labels": len(block["observations"]),
                "cumulative_labels": len(aggregate_observations),
                "training_failures": int(np.sum(block["terminated"])),
                "training_segments": int(np.max(block["segment_ids"]) + 1),
                "preupdate_learner_expert_disagreement_rate": disagreement,
                "evaluation": evaluation,
            }
        )
    return log


def run_smile(args, run_directory: Path) -> list[dict]:
    method_directory = run_directory / "smile"
    data_directory = method_directory / "data"
    checkpoint_directory = method_directory / "checkpoints"
    data_directory.mkdir(parents=True)
    checkpoint_directory.mkdir()
    base_policies = []
    log = []

    for iteration in range(1, args.iterations + 1):
        environment_seed, selector_seed = make_iteration_seeds(
            args.seed, iteration, 2
        )
        collection_selector = smile_collection_selector(base_policies, args.alpha)
        block = collect_training_block(
            collection_selector,
            environment_seed,
            selector_seed,
            args.labels_per_iteration,
        )
        np.savez_compressed(
            data_directory / f"iteration_{iteration:03d}.npz", **block
        )
        # At iteration 1, use the same learner initialization as the paired
        # supervised and DAgger runs. Later SMILe components need independent
        # (but deterministic) initializations because each is a fresh policy.
        base_seed = args.seed + (iteration - 1) * 10_000
        base_trainer = make_bc_trainer(base_seed)
        fit_bc(
            base_trainer,
            block["observations"],
            block["expert_actions"],
            args.bc_epochs,
        )
        base_policies.append(base_trainer.policy)
        torch.save(
            base_trainer.policy.state_dict(),
            checkpoint_directory / f"base_policy_{iteration:03d}.pt",
        )
        evaluation = evaluate_selector(
            smile_expert_free_selector(base_policies, args.alpha),
            args.evaluation_seed,
            300_000 + args.seed * 1_000,
            args.eval_episodes,
        )
        collection_expert_fraction = float(np.mean(block["controller_ids"] == -1))
        theoretical_expert_fraction = (1.0 - args.alpha) ** (iteration - 1)
        log.append(
            {
                "iteration": iteration,
                "alpha": args.alpha,
                "theoretical_collection_expert_fraction": (
                    theoretical_expert_fraction
                ),
                "realized_collection_expert_fraction": collection_expert_fraction,
                "new_labels": len(block["observations"]),
                "cumulative_labels": iteration * len(block["observations"]),
                "training_failures": int(np.sum(block["terminated"])),
                "training_segments": int(np.max(block["segment_ids"]) + 1),
                "learned_policy_component_count": len(base_policies),
                "evaluation": evaluation,
            }
        )
    return log


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--labels-per-iteration", type=int, default=500)
    parser.add_argument("--bc-epochs", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--evaluation-seed", type=int, default=40_000)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name")
    args = parser.parse_args()
    for name in (
        "iterations",
        "labels_per_iteration",
        "bc_epochs",
        "eval_episodes",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not 0.0 < args.alpha <= 1.0:
        parser.error("--alpha must be in (0,1]")
    return args


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    run_name = args.run_name or f"seed_{args.seed}"
    run_directory = args.output_root.resolve() / run_name
    if run_directory.exists():
        raise SystemExit(
            f"Refusing to overwrite existing run directory: {run_directory}"
        )
    run_directory.mkdir(parents=True)

    expert_evaluation = evaluate_selector(
        lambda observation, rng: (rule_expert(observation), 0),
        args.evaluation_seed,
        300_000 + args.seed * 1_000,
        args.eval_episodes,
    )
    print("Running supervised baseline", flush=True)
    supervised_log = run_supervised(args, run_directory)
    print("Running DAgger", flush=True)
    dagger_log = run_dagger(args, run_directory)
    print("Running SMILe", flush=True)
    smile_log = run_smile(args, run_directory)

    results = {
        "environment": "CartPole-v1",
        "seed": args.seed,
        "source_paper": (
            "Ross, Gordon, and Bagnell (2011), Super Tux Kart protocol adaptation"
        ),
        "protocol": {
            "iterations": args.iterations,
            "labels_per_iteration": args.labels_per_iteration,
            "evaluation_episodes_per_checkpoint": args.eval_episodes,
            "evaluation_seed": args.evaluation_seed,
            "bc_epochs_per_iteration": args.bc_epochs,
            "learner": "imitation FeedForward32Policy [32,32]",
            "dagger_beta": "1 at iteration 1; 0 thereafter",
            "smile_alpha": args.alpha,
            "supervised_collection": "rule expert only",
            "failure_definition": (
                "CartPole terminated before the 500-step TimeLimit"
            ),
        },
        "expert": expert_evaluation,
        "methods": {
            "supervised": supervised_log,
            "dagger": dagger_log,
            "smile": smile_log,
        },
        "versions": {
            "numpy": np.__version__,
            "gymnasium": gym.__version__,
            "torch": torch.__version__,
            "stable_baselines3": stable_baselines3.__version__,
            "imitation": imitation.__version__,
        },
    }
    results_path = run_directory / "results.json"
    results_path.write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(results, indent=2, sort_keys=True))
    print(f"Persistent run artifacts: {run_directory}")


if __name__ == "__main__":
    main()
