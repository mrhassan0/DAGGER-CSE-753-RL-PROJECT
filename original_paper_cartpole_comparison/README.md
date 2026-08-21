# Original-paper-style CartPole comparison

The original DAgger paper did not use CartPole. This folder transfers its Super
Tux Kart comparison protocol to the project's rule-expert CartPole task.

## Transferred settings

- 20 training iterations.
- 500 newly labeled CartPole states per method and iteration.
- DAgger mixing schedule `beta_i = 1[i = 1]`: expert control in the first
  iteration, learner control thereafter, with dense rule-expert labels.
- SMILe with `alpha = 0.1` and the stochastic policy-mixture update described
  by Ross, Gordon, and Bagnell (2011).
- Supervised baseline trained only on newly generated rule-expert trajectories.
- Performance checkpoints at iterations 1, 5, 10, 15, and 20.
- Five independent training seeds and two-sided 95% Student-t confidence
  intervals across seed means.

Super Tux Kart measured falls per lap. CartPole ends at its first failure, so
the closest lower-is-better analogue is the fraction of evaluation episodes
that terminate before 500 steps. Mean return is reported as a second metric.

## Run

From this folder:

```bash
../imitation_library_cartpole/.venv/bin/python run_five_seed_study.py
MPLCONFIGDIR=/tmp/matplotlib-rl-project \
  ../imitation_library_cartpole/.venv/bin/python analyze_study.py
```

The experiment is separate from both `../rule_based_cartpole/` and
`../imitation_library_cartpole/`; their existing results are not overwritten.

## Completed outputs

The five-seed study is complete under
`outputs/paper_style_cartpole_v1/`. Raw results, all 300 saved checkpoints,
and all 300 exact-size training blocks are under `runs/seed_0/` through
`runs/seed_4/`. The `analysis/` folder contains:

- paper-matched checkpoint graphs in PDF and PNG formats;
- full 20-iteration graphs in PDF and PNG formats;
- per-seed results in `per_seed_results.csv`;
- seed-aggregate means and 95% confidence intervals in
  `iteration_aggregate.csv` and `aggregate_results.json`.

Read `METHODOLOGY.md` for the exact protocol and `RESULTS.md` for the measured
comparison and its interpretation.
