# Library-based CartPole DAgger baseline

This is an independent comparison implementation using the third-party
imitation package's SimpleDAggerTrainer.

It is deliberately separated from ../datasets/cartpole/:

- The main dataset uses the rule-based expert specified in the project PDF.
- This baseline first trains a PPO expert and then lets the library generate,
  aggregate, and save DAgger demonstrations online.
- It uses CartPole-v1, with the same 500-step horizon as the project plan.

Nothing in this folder is used by the rule-expert CSV pipeline unless a later
experiment explicitly selects it.

## Install

Use a virtual environment. The requirements select CPU-only PyTorch:

    python3 -m venv .venv
    .venv/bin/python -m pip install -r requirements.txt

## Run

    .venv/bin/python run_library_dagger.py

The defaults train a PPO expert for 50,000 environment steps, train DAgger for
8,000 environment steps, and evaluate both policies over 20 episodes.

The round logger uses the same evaluation seed at every checkpoint. This makes
round-to-round changes attributable to the learned policy instead of changing
the evaluation initial conditions.

For a quick smoke test:

    .venv/bin/python run_library_dagger.py --expert-timesteps 5000 --dagger-timesteps 1000 --eval-episodes 5 --run-name smoke

## Persistent outputs

Every run gets its own directory:

    outputs/<run-name>/
    ├── expert/
    │   └── ppo_cartpole_expert.zip
    ├── dagger/
    │   ├── demos/
    │   │   ├── round-000/
    │   │   ├── round-001/
    │   │   └── ...
    │   └── dagger_policy_state_dict.pt
    └── results.json

Unlike TemporaryDirectory, these generated demonstrations and policy weights
remain available for later analysis. Reusing an existing run name is rejected
to prevent accidental overwrites.

The complete library trainer object is not serialized because imitation 1.0.1
contains an unpicklable schedule closure under this Python stack. The runner
therefore saves the learned policy state dictionary, the full configuration,
the PPO expert model, and every generated demonstration trajectory.

## Five-seed reproducibility study

Run five complete experiments (new expert, learner, rollouts, and training for
each seed):

    .venv/bin/python run_five_seed_study.py

Aggregate the raw results and regenerate the graph:

    MPLCONFIGDIR=/tmp/matplotlib-rl-project \
      .venv/bin/python analyze_multiseed_study.py

The completed measured study is in:

    outputs/multiseed_fixed_eval_v1/

Its main report is:

    outputs/multiseed_fixed_eval_v1/report/five_seed_dagger_report.pdf

The measured round-0 mean was 191.29 with across-seed sample SD 97.19 and a
95% Student-t interval of [70.61, 311.97]. All five training seeds reached the
500-step ceiling at round 1 and stayed there through round 5. These results are
for the PPO-expert library baseline only, not the main project's rule-based
expert, OCR experiment, or novelty ablations.

## Important scope difference

This is a standard, dense-supervision DAgger reference baseline. The library
queries the PPO expert on every visited state. The project's planned
label-budgeted, periodic-query, uncertainty-query, retention, and OCR ablations
will require the custom project implementation.
