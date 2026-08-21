#!/usr/bin/env python3
"""Train a PPO expert and a persistent SimpleDAggerTrainer CartPole baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import imitation
import numpy as np
import stable_baselines3
import torch
from datasets import load_from_disk
from imitation.algorithms import bc
from imitation.algorithms.dagger import (
    ExponentialBetaSchedule,
    SimpleDAggerTrainer,
)
from imitation.data import rollout
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "outputs"


def make_vector_environment(seed: int) -> DummyVecEnv:
    """Construct one seeded CartPole-v1 environment."""

    def make_environment():
        environment = Monitor(gym.make("CartPole-v1"))
        environment.reset(seed=seed)
        environment.action_space.seed(seed)
        return environment

    return DummyVecEnv([make_environment])


def evaluate(policy, seed: int, episodes: int) -> dict:
    """Evaluate a policy alone on a reproducible sequence of episodes."""

    environment = make_vector_environment(seed)
    episode_rewards, episode_lengths = evaluate_policy(
        policy,
        environment,
        n_eval_episodes=episodes,
        deterministic=True,
        return_episode_rewards=True,
    )
    environment.close()
    rewards = np.asarray(episode_rewards, dtype=np.float64)
    return {
        "seed": seed,
        "episodes": episodes,
        "mean_reward": float(np.mean(rewards)),
        "reward_standard_deviation": float(np.std(rewards, ddof=0)),
        "episode_rewards": [float(value) for value in episode_rewards],
        "episode_lengths": [int(value) for value in episode_lengths],
    }


def train_dagger_with_round_log(
    dagger_trainer: SimpleDAggerTrainer,
    total_timesteps: int,
    eval_seed: int,
    eval_episodes: int,
    rollout_round_min_episodes: int,
    rollout_round_min_timesteps: int,
    bc_train_kwargs: dict,
) -> list[dict]:
    """Run SimpleDAggerTrainer's round loop by hand so we can evaluate the
    learner in isolation (no expert mixed in) after every round.

    Mirrors imitation.algorithms.dagger.SimpleDAggerTrainer.train(); the only
    addition is the per-round evaluate() call and round_log entry.
    """
    round_log = []
    total_timestep_count = 0
    round_num = 0

    while total_timestep_count < total_timesteps:
        beta = dagger_trainer.beta_schedule(round_num)
        collector = dagger_trainer.create_trajectory_collector()
        sample_until = rollout.make_sample_until(
            min_timesteps=max(rollout_round_min_timesteps, dagger_trainer.batch_size),
            min_episodes=rollout_round_min_episodes,
        )
        trajectories = rollout.generate_trajectories(
            policy=dagger_trainer.expert_policy,
            venv=collector,
            sample_until=sample_until,
            deterministic_policy=True,
            rng=collector.rng,
        )
        round_timestep_count = sum(len(traj) for traj in trajectories)
        total_timestep_count += round_timestep_count

        dagger_trainer.extend_and_update(bc_train_kwargs)

        learner_evaluation = evaluate(
            dagger_trainer.policy,
            seed=eval_seed,
            episodes=eval_episodes,
        )
        round_log.append(
            {
                "round": round_num,
                "beta": beta,
                "round_timesteps": round_timestep_count,
                "cumulative_dagger_timesteps": total_timestep_count,
                "evaluation_seed": eval_seed,
                "evaluation_episodes": eval_episodes,
                "learner_mean_return": learner_evaluation["mean_reward"],
                "learner_return_std": learner_evaluation[
                    "reward_standard_deviation"
                ],
                "learner_episode_returns": learner_evaluation[
                    "episode_rewards"
                ],
            }
        )
        round_num += 1

    return round_log


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--expert-timesteps", type=int, default=50_000)
    parser.add_argument("--dagger-timesteps", type=int, default=8_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--beta-decay", type=float, default=0.5)
    parser.add_argument("--expert-eval-seed", type=int, default=10_000)
    parser.add_argument("--final-eval-seed", type=int, default=30_000)
    parser.add_argument("--round-eval-seed", type=int, default=40_000)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name")
    args = parser.parse_args()

    for name in ("expert_timesteps", "dagger_timesteps", "eval_episodes"):
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

    run_name = args.run_name or (
        f"seed_{args.seed}_expert_{args.expert_timesteps}"
        f"_dagger_{args.dagger_timesteps}"
    )
    run_directory = args.output_root.resolve() / run_name
    if run_directory.exists():
        raise SystemExit(
            f"Refusing to overwrite existing run directory: {run_directory}"
        )

    expert_directory = run_directory / "expert"
    dagger_directory = run_directory / "dagger"
    expert_directory.mkdir(parents=True)
    dagger_directory.mkdir()

    expert_environment = make_vector_environment(args.seed)
    expert = PPO(
        "MlpPolicy",
        expert_environment,
        seed=args.seed,
        device="cpu",
        verbose=0,
    )
    expert.learn(total_timesteps=args.expert_timesteps, progress_bar=False)
    expert_path = expert_directory / "ppo_cartpole_expert"
    expert.save(expert_path)

    expert_evaluation = evaluate(
        expert,
        seed=args.expert_eval_seed,
        episodes=args.eval_episodes,
    )

    dagger_environment = make_vector_environment(args.seed + 20_000)
    bc_trainer = bc.BC(
        observation_space=dagger_environment.observation_space,
        action_space=dagger_environment.action_space,
        rng=rng,
        device="cpu",
    )
    dagger_trainer = SimpleDAggerTrainer(
        venv=dagger_environment,
        scratch_dir=dagger_directory,
        expert_policy=expert,
        bc_trainer=bc_trainer,
        beta_schedule=ExponentialBetaSchedule(args.beta_decay),
        rng=rng,
    )
    round_log = train_dagger_with_round_log(
        dagger_trainer,
        args.dagger_timesteps,
        eval_seed=args.round_eval_seed,
        eval_episodes=args.eval_episodes,
        rollout_round_min_episodes=3,
        rollout_round_min_timesteps=500,
        bc_train_kwargs={"n_epochs": 4, "progress_bar": False},
    )

    dagger_evaluation = evaluate(
        dagger_trainer.policy,
        seed=args.final_eval_seed,
        episodes=args.eval_episodes,
    )
    policy_weights_path = dagger_directory / "dagger_policy_state_dict.pt"
    torch.save(dagger_trainer.policy.state_dict(), policy_weights_path)
    dagger_environment.close()
    expert_environment.close()

    demonstration_directories = sorted(
        path
        for path in dagger_directory.glob("demos/round-*/*.npz")
        if path.is_dir()
    )
    demonstration_paths = [
        str(path.relative_to(run_directory))
        for path in demonstration_directories
    ]
    demonstration_timesteps = sum(
        len(load_from_disk(str(path))[0]["acts"])
        for path in demonstration_directories
    )
    results = {
        "environment": "CartPole-v1",
        "seed": args.seed,
        "derived_seeds": {
            "ppo_training": args.seed,
            "ppo_evaluation": args.expert_eval_seed,
            "dagger_training": args.seed + 20_000,
            "dagger_evaluation": args.final_eval_seed,
            "dagger_round_evaluation": args.round_eval_seed,
        },
        "evaluation_protocol": {
            "deterministic_actions": True,
            "common_expert_evaluation_seed_across_training_seeds": (
                args.expert_eval_seed
            ),
            "common_final_evaluation_seed_across_training_seeds": (
                args.final_eval_seed
            ),
            "common_round_evaluation_seed_across_rounds_and_training_seeds": (
                args.round_eval_seed
            ),
            "note": (
                "A fresh environment is reconstructed from the same seed for "
                "every checkpoint, so all checkpoints face the same deterministic "
                "sequence of initial conditions."
            ),
        },
        "expert": {
            "algorithm": "PPO",
            "requested_training_timesteps": args.expert_timesteps,
            "actual_training_timesteps": int(expert.num_timesteps),
            "mean_reward": expert_evaluation["mean_reward"],
            "reward_standard_deviation": expert_evaluation[
                "reward_standard_deviation"
            ],
            "evaluation_episode_rewards": expert_evaluation[
                "episode_rewards"
            ],
            "model": str(expert_path.with_suffix(".zip").relative_to(run_directory)),
        },
        "dagger": {
            "implementation": "imitation.algorithms.dagger.SimpleDAggerTrainer",
            "requested_training_timesteps": args.dagger_timesteps,
            "actual_demonstration_timesteps": demonstration_timesteps,
            "beta_schedule": "exponential",
            "beta_decay": args.beta_decay,
            "mean_reward": dagger_evaluation["mean_reward"],
            "reward_standard_deviation": dagger_evaluation[
                "reward_standard_deviation"
            ],
            "evaluation_episode_rewards": dagger_evaluation[
                "episode_rewards"
            ],
            "demonstration_trajectory_count": len(demonstration_paths),
            "demonstration_timesteps": demonstration_timesteps,
            "demonstration_directories": demonstration_paths,
            "round_log": round_log,
            "policy_state_dict": str(
                policy_weights_path.relative_to(run_directory)
            ),
        },
        "evaluation_episodes": args.eval_episodes,
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
