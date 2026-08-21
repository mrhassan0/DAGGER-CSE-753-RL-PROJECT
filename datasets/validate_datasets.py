#!/usr/bin/env python3
"""Validate the prepared Stanford OCR and CartPole datasets."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


DATASET_DIR = Path(__file__).resolve().parent
OCR_PATH = DATASET_DIR / "ocr" / "letter.data"
CARTPOLE_PATH = (
    DATASET_DIR / "cartpole" / "cartpole_expert_demonstrations.csv"
)
CARTPOLE_METADATA_PATH = CARTPOLE_PATH.with_suffix(".metadata.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_ocr() -> dict:
    row_count = 0
    terminal_rows = 0
    folds: Counter[int] = Counter()
    letters: Counter[str] = Counter()
    word_lengths: Counter[int] = Counter()

    with OCR_PATH.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            fields = line.rstrip("\n").rstrip("\t").split("\t")
            if len(fields) != 134:
                raise ValueError(
                    f"OCR line {line_number}: expected 134 fields, found {len(fields)}"
                )

            letter = fields[1]
            next_id = int(fields[2])
            position = int(fields[4])
            fold = int(fields[5])
            pixels = fields[6:]

            if letter not in "abcdefghijklmnopqrstuvwxyz":
                raise ValueError(f"OCR line {line_number}: invalid letter {letter!r}")
            if fold not in range(10):
                raise ValueError(f"OCR line {line_number}: invalid fold {fold}")
            if any(pixel not in {"0", "1"} for pixel in pixels):
                raise ValueError(f"OCR line {line_number}: pixels must be binary")

            row_count += 1
            folds[fold] += 1
            letters[letter] += 1
            if next_id == -1:
                terminal_rows += 1
                word_lengths[position] += 1

    if row_count != 52152 or terminal_rows != 6877:
        raise ValueError(
            f"Unexpected OCR size: {row_count} rows and {terminal_rows} words"
        )
    if len(letters) != 26 or len(folds) != 10:
        raise ValueError("OCR dataset must contain 26 classes and 10 folds")

    return {
        "rows": row_count,
        "words": terminal_rows,
        "classes": len(letters),
        "folds": len(folds),
        "minimum_word_length": min(word_lengths),
        "maximum_word_length": max(word_lengths),
        "sha256": sha256(OCR_PATH),
    }


def validate_cartpole() -> dict:
    required_fields = {
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
    }
    rows = 0
    episodes: Counter[int] = Counter()
    actions: Counter[int] = Counter()

    with CARTPOLE_PATH.open(newline="", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file)
        if set(reader.fieldnames or []) != required_fields:
            raise ValueError("CartPole CSV schema does not match the expected fields")

        for row in reader:
            episode_id = int(row["episode_id"])
            timestep = int(row["timestep"])
            action = int(row["expert_action"])
            if timestep != episodes[episode_id]:
                raise ValueError(
                    f"CartPole episode {episode_id}: unexpected timestep {timestep}"
                )
            if action not in {0, 1}:
                raise ValueError(f"CartPole row {rows + 2}: invalid action {action}")

            for field in (
                "cart_position",
                "cart_velocity",
                "pole_angle",
                "pole_angular_velocity",
                "reward",
            ):
                float(row[field])

            rows += 1
            episodes[episode_id] += 1
            actions[action] += 1

    metadata = json.loads(CARTPOLE_METADATA_PATH.read_text(encoding="utf-8"))
    checksum = sha256(CARTPOLE_PATH)
    if rows != metadata["transitions"]:
        raise ValueError("CartPole row count does not match its metadata")
    if checksum != metadata["sha256"]:
        raise ValueError("CartPole checksum does not match its metadata")

    return {
        "rows": rows,
        "episodes": len(episodes),
        "minimum_episode_length": min(episodes.values()),
        "maximum_episode_length": max(episodes.values()),
        "action_counts": dict(sorted(actions.items())),
        "backend": metadata["backend"],
        "sha256": checksum,
    }


def main() -> None:
    print("OCR dataset")
    print(json.dumps(validate_ocr(), indent=2, sort_keys=True))
    print("CartPole dataset")
    print(json.dumps(validate_cartpole(), indent=2, sort_keys=True))
    print("All dataset checks passed.")


if __name__ == "__main__":
    main()
