#!/usr/bin/env python3
"""Run one CartPole DAgger experiment using the project PDF's rule expert."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import gymnasium as gym
import imitation
import numpy as np
import stable_baselines3
import torch
from imitation.algorithms import bc
from imitation.data import types


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_DATASET = (
    PROJECT_DIR / "datasets" / "cartpole" / "cartpole_expert_demonstrations.csv"
)
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "outputs" / "rule_multiseed_fixed_eval_v1" / "runs"


def rule_expert(observation: np.ndarray) -> int:
    """Return the exact rule-based action specified in the project PDF."""

    _, cart_velocity, pole_angle, pole_angular_velocity = observation
    score = pole_angle + 0.6 * pole_angular_velocity + 0.02 * cart_velocity
    return int(score > 0.0)


def dataset_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as dataset_file:
        for chunk in iter(lambda: dataset_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_initial_demonstrations(
    path: Path,
    episode_count: int,
) -> dict[str, np.ndarray]:
    """Load complete episodes 0..episode_count-1 from the generated CSV."""

    observations = []
    expert_actions = []
    episode_ids = []
    timesteps = []
    with path.open(newline="", encoding="utf-8") as dataset_file:
        reader = csv.DictReader(dataset_file)
        for row in reader:
            episode_id = int(row["episode_id"])
            if episode_id >= episode_count:
                continue
            observation = np.asarray(
                [
                    float(row["cart_position"]),
                    float(row["cart_velocity"]),
                    float(row["pole_angle"]),
                    float(row["pole_angular_velocity"]),
                ],
                dtype=np.float32,
            )
            recorded_action = int(row["expert_action"])
            computed_action = rule_expert(observation)
            if recorded_action != computed_action:
                raise ValueError(
                    "CSV expert action does not match the PDF rule at "
                    f"episode {episode_id}, timestep {row['timestep']}"
                )
            observations.append(observation)
            expert_actions.append(recorded_action)
            episode_ids.append(episode_id)
            timesteps.append(int(row["timestep"]))

    if not observations:
        raise ValueError(f"No initial demonstrations loaded from {path}")
    observed_episode_ids = sorted(set(episode_ids))
    expected_episode_ids = list(range(episode_count))
    if observed_episode_ids != expected_episode_ids:
        raise ValueError(
            f"Expected complete episode IDs {expected_episode_ids}, found "
            f"{observed_episode_ids}"
        )
    for episode_id in expected_episode_ids:
        episode_steps = [
            step for step, identifier in zip(timesteps, episode_ids)
            if identifier == episode_id
        ]
        if episode_steps != list(range(len(episode_steps))):
            raise ValueError(f"Episode {episode_id} has non-contiguous timesteps")

    return {
        "observations": np.asarray(observations, dtype=np.float32),
        "expert_actions": np.asarray(expert_actions, dtype=np.int64),
        "episode_ids": np.asarray(episode_ids, dtype=np.int64),
        "timesteps": np.asarray(timesteps, dtype=np.int64),
    }


def make_bc_transitions(
    observations: np.ndarray,
    expert_actions: np.ndarray,
) -> types.Transitions:
    """Create the transition container expected by imitation 1.0.1's BC loader.

    BC optimizes only observations and expert actions. The package collator
    nevertheless requires next_obs and dones fields, so next_obs is a shape-
    compatible copy and dones is false. Neither field enters the BC loss.
    """

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


def learner_action(policy, observation: np.ndarray) -> int:
    action, _ = policy.predict(observation, deterministic=True)
    return int(np.asarray(action).item())


def evaluate(policy, seed: int, episodes: int) -> dict:
    """Evaluate a learner alone on the baseline's fixed episode sequence."""

    environment = gym.make("CartPole-v1")
    environment.reset(seed=seed)
    environment.action_space.seed(seed)
    # The PPO-library baseline seeds once in its environment factory and then
    # resets again inside evaluate_policy. Repeating that reset here produces
    # the same initial-condition sequence for a direct comparison.
    observation, _ = environment.reset()
    episode_returns = []
    episode_lengths = []
    for episode_index in range(episodes):
        episode_return = 0.0
        episode_length = 0
        while True:
            action = learner_action(policy, observation)
            observation, reward, terminated, truncated, _ = environment.step(action)
            episode_return += float(reward)
            episode_length += 1
            if terminated or truncated:
                break
        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)
        if episode_index + 1 < episodes:
            observation, _ = environment.reset()
    environment.close()
    rewards = np.asarray(episode_returns, dtype=np.float64)
    return {
        "seed": seed,
        "episodes": episodes,
        "mean_reward": float(np.mean(rewards)),
        "reward_standard_deviation": float(np.std(rewards, ddof=0)),
        "episode_rewards": episode_returns,
        "episode_lengths": episode_lengths,
    }


def evaluate_rule_expert(seed: int, episodes: int) -> dict:
    environment = gym.make("CartPole-v1")
    environment.reset(seed=seed)
    environment.action_space.seed(seed)
    observation, _ = environment.reset()
    episode_returns = []
    episode_lengths = []
    for episode_index in range(episodes):
        episode_return = 0.0
        episode_length = 0
        while True:
            observation, reward, terminated, truncated, _ = environment.step(
                rule_expert(observation)
            )
            episode_return += float(reward)
            episode_length += 1
            if terminated or truncated:
                break
        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)
        if episode_index + 1 < episodes:
            observation, _ = environment.reset()
    environment.close()
    rewards = np.asarray(episode_returns, dtype=np.float64)
    return {
        "seed": seed,
        "episodes": episodes,
        "mean_reward": float(np.mean(rewards)),
        "reward_standard_deviation": float(np.std(rewards, ddof=0)),
        "episode_rewards": episode_returns,
        "episode_lengths": episode_lengths,
    }


def collect_online_round(
    environment: gym.Env,
    policy,
    rng: np.random.Generator,
    beta: float,
    minimum_episodes: int,
    minimum_timesteps: int,
) -> dict[str, np.ndarray]:
    """Collect complete mixed-policy episodes and label every visited state."""

    observations = []
    expert_actions = []
    learner_actions = []
    executed_actions = []
    expert_controlled = []
    rewards = []
    terminated_flags = []
    truncated_flags = []
    episode_ids = []
    timesteps = []
    episode_count = 0

    while episode_count < minimum_episodes or len(observations) < minimum_timesteps:
        observation, _ = environment.reset()
        timestep = 0
        while True:
            observation_array = np.asarray(observation, dtype=np.float32)
            expert_action = rule_expert(observation_array)
            predicted_action = learner_action(policy, observation_array)
            use_expert = bool(rng.random() < beta)
            executed_action = expert_action if use_expert else predicted_action
            next_observation, reward, terminated, truncated, _ = environment.step(
                executed_action
            )

            observations.append(observation_array)
            expert_actions.append(expert_action)
            learner_actions.append(predicted_action)
            executed_actions.append(executed_action)
            expert_controlled.append(use_expert)
            rewards.append(float(reward))
            terminated_flags.append(bool(terminated))
            truncated_flags.append(bool(truncated))
            episode_ids.append(episode_count)
            timesteps.append(timestep)

            observation = next_observation
            timestep += 1
            if terminated or truncated:
                break
        episode_count += 1

    return {
        "observations": np.asarray(observations, dtype=np.float32),
        "expert_actions": np.asarray(expert_actions, dtype=np.int64),
        "learner_actions": np.asarray(learner_actions, dtype=np.int64),
        "executed_actions": np.asarray(executed_actions, dtype=np.int64),
        "expert_controlled": np.asarray(expert_controlled, dtype=bool),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "terminated": np.asarray(terminated_flags, dtype=bool),
        "truncated": np.asarray(truncated_flags, dtype=bool),
        "episode_ids": np.asarray(episode_ids, dtype=np.int64),
        "timesteps": np.asarray(timesteps, dtype=np.int64),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--initial-episodes", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--beta-decay", type=float, default=0.5)
    parser.add_argument("--minimum-round-episodes", type=int, default=3)
    parser.add_argument("--minimum-round-timesteps", type=int, default=500)
    parser.add_argument("--bc-epochs", type=int, default=4)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--expert-eval-seed", type=int, default=10_000)
    parser.add_argument("--final-eval-seed", type=int, default=30_000)
    parser.add_argument("--round-eval-seed", type=int, default=40_000)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name")
    args = parser.parse_args()
    for name in (
        "initial_episodes",
        "rounds",
        "minimum_round_episodes",
        "minimum_round_timesteps",
        "bc_epochs",
        "eval_episodes",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not 0.0 < args.beta_decay <= 1.0:
        parser.error("--beta-decay must be in (0, 1]")
    return args


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    dataset_path = args.dataset.resolve()
    run_name = args.run_name or f"seed_{args.seed}"
    run_directory = args.output_root.resolve() / run_name
    if run_directory.exists():
        raise SystemExit(
            f"Refusing to overwrite existing run directory: {run_directory}"
        )
    data_directory = run_directory / "round_data"
    checkpoint_directory = run_directory / "checkpoints"
    data_directory.mkdir(parents=True)
    checkpoint_directory.mkdir()

    initial = load_initial_demonstrations(dataset_path, args.initial_episodes)
    np.savez_compressed(
        data_directory / "round_000_initial_csv.npz",
        observations=initial["observations"],
        expert_actions=initial["expert_actions"],
        episode_ids=initial["episode_ids"],
        timesteps=initial["timesteps"],
    )

    space_environment = gym.make("CartPole-v1")
    observation_space = space_environment.observation_space
    action_space = space_environment.action_space
    space_environment.close()
    bc_trainer = bc.BC(
        observation_space=observation_space,
        action_space=action_space,
        rng=rng,
        device="cpu",
    )

    all_observations = [initial["observations"]]
    all_expert_actions = [initial["expert_actions"]]
    round_log = []

    bc_trainer.set_demonstrations(
        make_bc_transitions(initial["observations"], initial["expert_actions"])
    )
    bc_trainer.train(n_epochs=args.bc_epochs, progress_bar=False)
    torch.save(
        bc_trainer.policy.state_dict(),
        checkpoint_directory / "round_000_policy_state_dict.pt",
    )
    round_zero_evaluation = evaluate(
        bc_trainer.policy,
        args.round_eval_seed,
        args.eval_episodes,
    )
    round_log.append(
        {
            "round": 0,
            "beta": 1.0,
            "data_source": "generated_rule_expert_csv_episodes_0_to_2",
            "round_timesteps": len(initial["observations"]),
            "round_episode_count": args.initial_episodes,
            "cumulative_dagger_labels": len(initial["observations"]),
            "realized_expert_control_fraction": 1.0,
            "learner_expert_disagreement_rate_before_update": None,
            "evaluation_seed": args.round_eval_seed,
            "evaluation_episodes": args.eval_episodes,
            "learner_mean_return": round_zero_evaluation["mean_reward"],
            "learner_return_std": round_zero_evaluation[
                "reward_standard_deviation"
            ],
            "learner_episode_returns": round_zero_evaluation["episode_rewards"],
        }
    )

    training_environment = gym.make("CartPole-v1")
    training_environment.reset(seed=args.seed + 20_000)
    training_environment.action_space.seed(args.seed + 20_000)

    for round_index in range(1, args.rounds):
        beta = args.beta_decay**round_index
        collected = collect_online_round(
            training_environment,
            bc_trainer.policy,
            rng,
            beta,
            args.minimum_round_episodes,
            args.minimum_round_timesteps,
        )
        data_path = data_directory / f"round_{round_index:03d}_online.npz"
        np.savez_compressed(data_path, **collected)
        disagreement_rate = float(
            np.mean(collected["learner_actions"] != collected["expert_actions"])
        )
        realized_expert_fraction = float(np.mean(collected["expert_controlled"]))

        all_observations.append(collected["observations"])
        all_expert_actions.append(collected["expert_actions"])
        aggregate_observations = np.concatenate(all_observations)
        aggregate_actions = np.concatenate(all_expert_actions)
        bc_trainer.set_demonstrations(
            make_bc_transitions(aggregate_observations, aggregate_actions)
        )
        bc_trainer.train(n_epochs=args.bc_epochs, progress_bar=False)
        torch.save(
            bc_trainer.policy.state_dict(),
            checkpoint_directory
            / f"round_{round_index:03d}_policy_state_dict.pt",
        )
        learner_evaluation = evaluate(
            bc_trainer.policy,
            args.round_eval_seed,
            args.eval_episodes,
        )
        round_log.append(
            {
                "round": round_index,
                "beta": beta,
                "data_source": "online_mixed_policy_rule_expert_labels",
                "round_timesteps": len(collected["observations"]),
                "round_episode_count": int(
                    np.max(collected["episode_ids"]) + 1
                ),
                "cumulative_dagger_labels": len(aggregate_observations),
                "realized_expert_control_fraction": realized_expert_fraction,
                "learner_expert_disagreement_rate_before_update": (
                    disagreement_rate
                ),
                "evaluation_seed": args.round_eval_seed,
                "evaluation_episodes": args.eval_episodes,
                "learner_mean_return": learner_evaluation["mean_reward"],
                "learner_return_std": learner_evaluation[
                    "reward_standard_deviation"
                ],
                "learner_episode_returns": learner_evaluation[
                    "episode_rewards"
                ],
            }
        )

    training_environment.close()
    final_evaluation = evaluate(
        bc_trainer.policy,
        args.final_eval_seed,
        args.eval_episodes,
    )
    expert_evaluation = evaluate_rule_expert(
        args.expert_eval_seed,
        args.eval_episodes,
    )
    final_policy_path = checkpoint_directory / "final_policy_state_dict.pt"
    torch.save(bc_trainer.policy.state_dict(), final_policy_path)

    results = {
        "environment": "CartPole-v1",
        "seed": args.seed,
        "expert": {
            "type": "deterministic_rule_based",
            "formula": "1[theta + 0.6*theta_dot + 0.02*x_dot > 0]",
            "mean_reward": expert_evaluation["mean_reward"],
            "reward_standard_deviation": expert_evaluation[
                "reward_standard_deviation"
            ],
            "evaluation_episode_rewards": expert_evaluation["episode_rewards"],
        },
        "initial_dataset": {
            "path": str(dataset_path),
            "sha256": dataset_sha256(dataset_path),
            "available_transitions": 50_000,
            "selected_episode_ids": list(range(args.initial_episodes)),
            "selected_transitions": len(initial["observations"]),
            "selection_reason": (
                "Three complete 500-step episodes match the PPO-library "
                "baseline's 1500-label round-0 budget."
            ),
        },
        "learner": {
            "implementation": "imitation.algorithms.bc.BC",
            "policy": "FeedForward32Policy with hidden layers [32, 32]",
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs_per_round": args.bc_epochs,
            "retention": "all aggregated data",
        },
        "dagger": {
            "rounds": args.rounds,
            "beta_schedule": "exponential",
            "beta_decay": args.beta_decay,
            "minimum_round_episodes": args.minimum_round_episodes,
            "minimum_round_timesteps": args.minimum_round_timesteps,
            "dense_expert_labels": True,
            "actual_total_labels": round_log[-1]["cumulative_dagger_labels"],
            "mean_reward": final_evaluation["mean_reward"],
            "reward_standard_deviation": final_evaluation[
                "reward_standard_deviation"
            ],
            "evaluation_episode_rewards": final_evaluation["episode_rewards"],
            "round_log": round_log,
            "final_policy_state_dict": str(
                final_policy_path.relative_to(run_directory)
            ),
        },
        "derived_seeds": {
            "learner_training": args.seed,
            "dagger_collection": args.seed + 20_000,
            "expert_evaluation": args.expert_eval_seed,
            "dagger_evaluation": args.final_eval_seed,
            "dagger_round_evaluation": args.round_eval_seed,
        },
        "evaluation_protocol": {
            "episodes": args.eval_episodes,
            "deterministic_learner_actions": True,
            "common_round_seed_across_rounds_and_training_seeds": (
                args.round_eval_seed
            ),
            "common_final_seed_across_training_seeds": args.final_eval_seed,
            "no_expert_control_during_evaluation": True,
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
