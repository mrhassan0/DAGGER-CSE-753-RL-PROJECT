# Rule-based CartPole DAgger

This folder is the primary CartPole path based on the expert formula in the
project PDF:

```text
expert_action = 1[theta + 0.6 * theta_dot + 0.02 * x_dot > 0]
```

It is separate from `../imitation_library_cartpole/`, which trains PPO experts.

## Method

- Round 0 takes three complete expert episodes (1,500 transitions) from the
  generated CSV in `../datasets/cartpole/`.
- Rounds 1--5 execute online mixed expert/learner rollouts in CartPole-v1.
- The rule expert labels every state visited during those online rollouts.
- All accumulated labels are retained, and the learner is retrained for four
  behavior-cloning epochs after every round.
- Five independently initialized learners are evaluated on the same fixed
  episode sequences used by the PPO-expert baseline.

Using 1,500 of the available 50,000 CSV rows matches the initial label count of
the PPO-library experiment. Training on all 50,000 rows would answer a different
question and would not be a label-matched comparison.

## Run

The existing library environment contains the required packages:

```bash
../imitation_library_cartpole/.venv/bin/python run_five_seed_study.py
MPLCONFIGDIR=/tmp/matplotlib-rl-project \
  ../imitation_library_cartpole/.venv/bin/python analyze_multiseed_study.py
```

To create a separate environment instead:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Every run has its own directory and refuses to overwrite an existing result.
Each round's collected data, policy checkpoint, evaluation returns, and summary
statistics are retained.

## Completed five-seed result

The measured rule-expert curve is:

- round 0: mean 175.89, sample SD 47.78, 95% Student-t CI
  [116.57, 235.21];
- round 1: mean 489.11, sample SD 7.21, 95% CI [480.16, 498.06];
- rounds 2--5: all five seeds scored 500.00.

The PPO-expert baseline reached 500 for all seeds at round 1. Both pipelines
ended at 500. The complete rule methodology and comparison are in:

```text
outputs/rule_multiseed_fixed_eval_v1/report/rule_expert_dagger_report.pdf
```
