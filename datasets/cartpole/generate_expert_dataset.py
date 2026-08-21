#!/usr/bin/env python3
"""Generate the initial CartPole expert-demonstration dataset.

The generator follows the CartPole MDP and rule-based expert specified in the
Final Project plan. It uses Gymnasium's CartPole-v1 when available. A small
dependency-free implementation with the same core physics and stopping rules is
provided so the dataset layer remains reproducible in a minimal environment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_OUTPUT = Path(__file__).with_name("cartpole_expert_demonstrations.csv")
CSV_FIELDS = [
    "episode_id",
    "timestep",
    "cart_position",
    "cart_velocity",
    "pole_angle",
    "pole_angular_velocity",
    "expert_action",
    "reward",
    "terminated",
    "truncated",
]


def expert_policy(state: Sequence[float]) -> int:
    """Rule-based expert from the project plan."""
    _, cart_velocity, pole_angle, pole_angular_velocity = state
    score = pole_angle + 0.6 * pole_angular_velocity + 0.02 * cart_velocity
    return int(score > 0.0)


class BuiltinCartPole:
    """Dependency-free CartPole-v1-compatible dynamics for data preparation."""

    gravity = 9.8
    mass_cart = 1.0
    mass_pole = 0.1
    total_mass = mass_cart + mass_pole
    half_pole_length = 0.5
    polemass_length = mass_pole * half_pole_length
    force_magnitude = 10.0
    timestep = 0.02
    cart_threshold = 2.4
    angle_threshold = 12.0 * 2.0 * math.pi / 360.0
    max_steps = 500

    def __init__(self) -> None:
        self.state = [0.0, 0.0, 0.0, 0.0]
        self.steps = 0

    def reset(self, seed: int) -> list[float]:
        rng = random.Random(seed)
        self.state = [rng.uniform(-0.05, 0.05) for _ in range(4)]
        self.steps = 0
        return self.state.copy()

    def step(self, action: int) -> tuple[list[float], float, bool, bool]:
        x, x_dot, theta, theta_dot = self.state
        force = self.force_magnitude if action == 1 else -self.force_magnitude
        cosine = math.cos(theta)
        sine = math.sin(theta)

        temporary = (
            force + self.polemass_length * theta_dot**2 * sine
        ) / self.total_mass
        theta_acceleration = (
            self.gravity * sine - cosine * temporary
        ) / (
            self.half_pole_length
            * (4.0 / 3.0 - self.mass_pole * cosine**2 / self.total_mass)
        )
        x_acceleration = (
            temporary
            - self.polemass_length
            * theta_acceleration
            * cosine
            / self.total_mass
        )

        x = x + self.timestep * x_dot
        x_dot = x_dot + self.timestep * x_acceleration
        theta = theta + self.timestep * theta_dot
        theta_dot = theta_dot + self.timestep * theta_acceleration

        self.state = [x, x_dot, theta, theta_dot]
        self.steps += 1

        terminated = abs(x) > self.cart_threshold or abs(theta) > self.angle_threshold
        truncated = self.steps >= self.max_steps and not terminated
        return self.state.copy(), 1.0, terminated, truncated


def select_backend(requested: str):
    if requested in {"auto", "gymnasium"}:
        try:
            import gymnasium as gym
        except ModuleNotFoundError:
            if requested == "gymnasium":
                raise SystemExit(
                    "Gymnasium is not installed. Install gymnasium[classic_control] "
                    "or use --backend builtin."
                )
        else:
            return "gymnasium", gym.make("CartPole-v1")

    return "builtin", BuiltinCartPole()


def reset_environment(backend: str, environment, seed: int) -> list[float]:
    if backend == "gymnasium":
        observation, _ = environment.reset(seed=seed)
        return [float(value) for value in observation]
    return environment.reset(seed)


def step_environment(backend: str, environment, action: int):
    if backend == "gymnasium":
        observation, reward, terminated, truncated, _ = environment.step(action)
        return (
            [float(value) for value in observation],
            float(reward),
            bool(terminated),
            bool(truncated),
        )
    return environment.step(action)


def format_float(value: float) -> str:
    return format(value, ".17g")


def write_dataset(
    output_path: Path,
    episodes: int,
    seed: int,
    requested_backend: str,
) -> dict:
    backend, environment = select_backend(requested_backend)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    episode_lengths: list[int] = []
    action_counts = {0: 0, 1: 0}

    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for episode_id in range(episodes):
            state = reset_environment(backend, environment, seed + episode_id)

            for timestep in range(500):
                action = expert_policy(state)
                action_counts[action] += 1
                next_state, reward, terminated, truncated = step_environment(
                    backend, environment, action
                )

                writer.writerow(
                    {
                        "episode_id": episode_id,
                        "timestep": timestep,
                        "cart_position": format_float(state[0]),
                        "cart_velocity": format_float(state[1]),
                        "pole_angle": format_float(state[2]),
                        "pole_angular_velocity": format_float(state[3]),
                        "expert_action": action,
                        "reward": format_float(reward),
                        "terminated": int(terminated),
                        "truncated": int(truncated),
                    }
                )

                state = next_state
                if terminated or truncated:
                    episode_lengths.append(timestep + 1)
                    break

    if backend == "gymnasium":
        environment.close()

    checksum = hashlib.sha256(output_path.read_bytes()).hexdigest()
    transition_count = sum(episode_lengths)
    metadata = {
        "schema_version": 1,
        "task": "CartPole-v1",
        "backend": backend,
        "requested_backend": requested_backend,
        "base_seed": seed,
        "episodes": episodes,
        "transitions": transition_count,
        "episode_length": {
            "minimum": min(episode_lengths),
            "maximum": max(episode_lengths),
            "mean": statistics.mean(episode_lengths),
            "solved_500_steps": sum(length == 500 for length in episode_lengths),
        },
        "expert_action_counts": {
            "push_left_0": action_counts[0],
            "push_right_1": action_counts[1],
        },
        "state_order": [
            "cart_position",
            "cart_velocity",
            "pole_angle",
            "pole_angular_velocity",
        ],
        "expert_policy": "1[theta + 0.6*theta_dot + 0.02*x_dot > 0]",
        "initial_state_distribution": "uniform(-0.05, 0.05)^4",
        "maximum_episode_steps": 500,
        "sha256": checksum,
    }

    metadata_path = output_path.with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def parse_args(arguments: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--backend",
        choices=("auto", "gymnasium", "builtin"),
        default="auto",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(arguments)
    if args.episodes <= 0:
        parser.error("--episodes must be a positive integer")
    return args


def main() -> None:
    args = parse_args()
    metadata = write_dataset(args.output, args.episodes, args.seed, args.backend)
    print(f"Generated {args.output}")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
