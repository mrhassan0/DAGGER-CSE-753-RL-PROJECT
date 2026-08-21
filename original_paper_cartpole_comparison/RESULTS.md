# Results: DAgger, SMILe, and supervised CartPole comparison

The study completed five independent training seeds, 20 iterations per method,
500 new expert labels per iteration, and 20 fixed evaluation episodes per
checkpoint. The table reports the mean over the five seed-level means and the
two-sided 95% Student-t confidence interval in brackets.

| Iteration | Labels/seed | Method | Failure rate | Mean return |
|---:|---:|:---|---:|---:|
| 1 | 500 | DAgger | 1.000 [1.000, 1.000] | 79.55 [72.94, 86.16] |
| 1 | 500 | SMILe | 1.000 [1.000, 1.000] | 79.55 [72.94, 86.16] |
| 1 | 500 | Supervised | 1.000 [1.000, 1.000] | 79.55 [72.94, 86.16] |
| 5 | 2,500 | DAgger | 0.030 [0.000, 0.064] | 496.65 [492.83, 500.00] |
| 5 | 2,500 | SMILe | 1.000 [1.000, 1.000] | 85.57 [74.28, 96.86] |
| 5 | 2,500 | Supervised | 0.000 [0.000, 0.000] | 500.00 [500.00, 500.00] |
| 10 | 5,000 | DAgger | 0.000 [0.000, 0.000] | 500.00 [500.00, 500.00] |
| 10 | 5,000 | SMILe | 1.000 [1.000, 1.000] | 97.78 [85.20, 110.36] |
| 10 | 5,000 | Supervised | 0.000 [0.000, 0.000] | 500.00 [500.00, 500.00] |
| 15 | 7,500 | DAgger | 0.000 [0.000, 0.000] | 500.00 [500.00, 500.00] |
| 15 | 7,500 | SMILe | 1.000 [1.000, 1.000] | 112.39 [99.83, 124.95] |
| 15 | 7,500 | Supervised | 0.000 [0.000, 0.000] | 500.00 [500.00, 500.00] |
| 20 | 10,000 | DAgger | 0.000 [0.000, 0.000] | 500.00 [500.00, 500.00] |
| 20 | 10,000 | SMILe | 1.000 [1.000, 1.000] | 119.55 [104.18, 134.92] |
| 20 | 10,000 | Supervised | 0.000 [0.000, 0.000] | 500.00 [500.00, 500.00] |

## Interpretation

All three curves are identical at iteration 1 because they use the same
expert-controlled data and paired learner initialization. DAgger improves very
quickly after it starts collecting learner-induced states: at iteration 5 it
has only 3 failures among the combined 100 evaluation episodes, and at all
paper checkpoints from iteration 10 onward every evaluated episode reaches the
500-step ceiling.

The supervised baseline is not weak here. From iteration 5 onward every seed
scores 500 on the fixed evaluation episodes. The CartPole expert is a simple
linear threshold in the observed state variables, so expert-distribution data
are sufficient for this neural classifier. Super Tux used image features and
recovery behavior that expert-only laps did not adequately cover; consequently
the paper's qualitative supervised result should not be expected automatically
on this easier task.

SMILe improves its return slowly but terminates before 500 in every evaluated
episode. Its expert-free evaluation policy is a stochastic, renormalized
mixture of all learned base policies. With `alpha=0.1`, early weak policies keep
substantial probability, and sampling a poor component during an episode can
destabilize the pole. This is the intended SMILe algorithm, not a plotting or
data-loading error.

These findings apply to the five learner seeds and the fixed 20-episode reset
sequence used here. A zero observed failure rate does not prove a zero failure
probability on every possible CartPole initial state.

## Main artifacts

- `analysis/paper_style_failure_comparison.pdf` and `.png`: closest analogue
  to the original paper's falls-per-lap figure.
- `analysis/paper_style_return_comparison.pdf` and `.png`: secondary mean-return
  comparison.
- `analysis/all_iterations_*`: diagnostic graphs showing every iteration.
- `analysis/per_seed_results.csv`: all seed/method/iteration measurements.
- `analysis/iteration_aggregate.csv`: means, sample standard deviations,
  standard errors, confidence bounds, and unclipped margins of error.
- `runs/seed_*/`: raw JSON, 300 policy checkpoints, and 300 collected data
  blocks for full reproducibility.
