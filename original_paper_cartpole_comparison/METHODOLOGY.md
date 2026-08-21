# Methodology: Super-Tux-style comparison on CartPole

## Scope

This is a controlled **adaptation**, not a reproduction of the Super Tux Kart
experiment in Ross, Gordon, and Bagnell (2011). The paper did not report a
CartPole experiment. Its comparison structure is transferred to this
project's existing rule-expert `CartPole-v1` setting.

## What is retained from the paper

- DAgger, SMILe, and supervised imitation learning are compared.
- Every method receives the same number of new expert labels per iteration.
- Training lasts 20 iterations.
- DAgger uses `beta_i = 1[i = 1]`: the expert controls iteration 1 and the
  current learner controls all later collection iterations.
- SMILe uses `alpha = 0.1`.
- The main figure uses checkpoints 1, 5, 10, 15, and 20 and plots performance
  against cumulative training data.
- Uncertainty is shown with two-sided 95% confidence intervals.

## CartPole-specific choices

- Environment: Gymnasium `CartPole-v1`, whose episode limit is 500 steps.
- Observation: cart position, cart velocity, pole angle, and pole angular
  velocity.
- Actions: `0` (push left) or `1` (push right).
- Rule expert:

  ```text
  action = 1[theta + 0.6 theta_dot + 0.02 x_dot > 0]
  ```

- Label budget: exactly 500 newly visited states per method, seed, and
  iteration. If an episode fails inside a collection block, the environment
  resets and collection continues until 500 states have been saved.
- Learner: the project's `imitation` `FeedForward32Policy` with two hidden
  layers of 32 units, Adam, batch size 32, and four BC epochs per update.

The paper's Super Tux setup instead used a human expert, continuous steering,
image features, linear ridge regression, and approximately 1,000 data points
from one lap per iteration. Those task-dependent details cannot be identical
on CartPole.

## Algorithms

### Supervised baseline

At every iteration the rule expert controls data collection. The new 500
state/action pairs are added to all earlier expert-controlled pairs, and the
same policy is trained again on the complete aggregate. It therefore sees
more data but never sees the state distribution caused by its own mistakes.

### DAgger

At iteration 1 the rule expert controls collection. From iteration 2 onward,
the current learner controls collection. The rule expert still labels every
visited state, including the learner's error and recovery states. Each update
trains on the aggregate of all blocks collected so far.

### SMILe

Before iteration `n`, data are collected with the stochastic policy mixture
from iteration `n-1`. A fresh base policy is trained only on that iteration's
500 labeled states. Its mixture weight is

```text
alpha (1 - alpha)^(n - 1),  where alpha = 0.1.
```

The remaining expert weight during collection is `(1 - alpha)^n`. For
evaluation, the expert component is removed and all learned-policy weights are
renormalized, exactly as defined in Section 2.3 of the paper. The mixture
selects a component independently at each step.

## Randomness and evaluation

- Independent training seeds: 0, 1, 2, 3, and 4.
- All methods use the same collection-environment seed within each paired
  training seed and iteration. Method-specific selector streams prevent
  stochastic policy choices from sharing accidental state.
- All checkpoints use the same fixed Gymnasium evaluation reset sequence,
  beginning with environment seed 40,000.
- Every method is evaluated for 20 episodes per checkpoint without expert
  control. This gives 100 episode outcomes per method/checkpoint across the
  five seeds.
- The seed is repeated for the complete training procedure; it is not merely
  five random starting observations.

## Metrics and confidence intervals

Super Tux measured falls per lap. A CartPole episode stops on its first fall,
so the closest lower-is-better analogue is

```text
failure rate = episodes terminated before step 500 / 20.
```

Mean episode return, whose maximum is 500, is reported as a secondary metric.
For each method and checkpoint, the five seed-level means are summarized using

```text
mean +/- t_(0.975, 4) * sample_SD / sqrt(5),
```

where `t_(0.975, 4) = 2.776`. Intervals are clipped only to the feasible plot
range: `[0, 1]` for failure rate and `[0, 500]` for return. The raw, unclipped
margin of error is retained in the aggregate CSV and JSON files.
