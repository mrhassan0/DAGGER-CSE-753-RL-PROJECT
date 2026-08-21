# Final Project

This is the clean implementation workspace for **Experimental Analysis of
DAgger**, following the supplied project plan in
`../Additional Files/main folder/main (1).pdf`.

Reproducible OCR and CartPole datasets are prepared. The primary rule-expert
CartPole DAgger experiment and a separate PPO-expert library comparison have
both been run for five independent training seeds. A separate 20-iteration,
five-seed CartPole comparison of DAgger, SMILe, and supervised imitation has
also been completed using the Super Tux experiment structure from the original
DAgger paper. The planned OCR algorithms, label-budget novelty, retention
ablations, and partial-observability experiments are not implemented yet.

## Dataset layout

```text
datasets/
├── ocr/
│   ├── letter.data
│   └── README.md
├── cartpole/
│   ├── cartpole_expert_demonstrations.csv
│   ├── cartpole_expert_demonstrations.metadata.json
│   ├── generate_expert_dataset.py
│   └── README.md
├── README.md
├── manifest.json
└── validate_datasets.py
requirements-datasets.txt
```

- **OCR** uses the full Stanford OCR handwritten-word dataset. Folds 0--8 are
  reserved for training and fold 9 for final testing, as specified in the
  project plan.
- **CartPole** uses expert state-action trajectories generated from
  `CartPole-v1`. The supplied generator uses Gymnasium when available and a
  dependency-free compatible simulator otherwise.

## Validate the prepared data

From this directory, run:

```bash
python3 datasets/validate_datasets.py
```

## Regenerate the CartPole demonstrations

```bash
python3 datasets/cartpole/generate_expert_dataset.py \
  --episodes 100 \
  --seed 0 \
  --backend auto
```

The committed CSV was generated with Gymnasium 1.3.0. Install the pinned
dataset dependency before regenerating it with that backend:

```bash
python3 -m pip install -r requirements-datasets.txt
python3 datasets/cartpole/generate_expert_dataset.py --backend gymnasium
```

The generated CartPole file is the initial expert/behavior-cloning dataset. The
rule-based DAgger implementation in `rule_based_cartpole/` collects additional
expert labels online from states visited by the learner; those learner-induced
samples are deliberately not pre-generated here.

## Primary rule-expert CartPole experiment

The `rule_based_cartpole/` folder initializes DAgger from three complete
episodes (1,500 rows) of the generated CSV, queries the PDF's rule expert on
online learner-influenced states, and retains the full aggregate. It includes
five raw runs, saved round data and policy checkpoints, statistical summaries,
rule-only and rule-versus-PPO graphs, and a separate one-column LaTeX report:

```text
rule_based_cartpole/outputs/rule_multiseed_fixed_eval_v1/
├── runs/                  # seed 0--4 rule-expert DAgger artifacts
├── logs/
├── analysis/              # CSV/JSON and both PDF/PNG graphs
└── report/
    ├── rule_expert_dagger_report.tex
    └── rule_expert_dagger_report.pdf
```

Measured five-seed rule-expert means were 175.89 at round 0, 489.11 at
round 1, and 500.00 from round 2 onward. Every final learner scored 500 on a
separate fixed evaluation sequence. The PPO-expert baseline reached the ceiling
one round earlier, at round 1; the rule report explains the comparison limits.

## Separate library-based comparison

The imitation_library_cartpole/ folder contains an independent
SimpleDAggerTrainer baseline. It trains a PPO expert and persistently stores the
library-generated DAgger trajectories and checkpoints. It is not mixed with the
rule-expert CSV dataset.

The completed five-seed baseline includes raw results, aggregate CSV/JSON,
saved models and demonstrations, a measured learning-curve graph, and a
one-column LaTeX report:

```text
imitation_library_cartpole/outputs/multiseed_fixed_eval_v1/
├── runs/                  # complete seed 0--4 artifacts
├── logs/                  # one execution log per seed
├── analysis/              # raw/aggregate tables and PDF/PNG graph
└── report/
    ├── five_seed_dagger_report.tex
    └── five_seed_dagger_report.pdf
```

The library baseline's measured round-0 mean was 191.29 (sample SD 97.19;
95% Student-t CI [70.61, 311.97]). Every seed reached return 500 at round 1
and stayed at the ceiling through round 5. The report explains the fixed-seed
evaluation methodology, the 9,000-versus-8,000 DAgger-step accounting, and the
limits of applying this result to the main project.

## Original-paper-style three-method comparison

The `original_paper_cartpole_comparison/` folder transfers the Super Tux Kart
comparison structure to rule-expert `CartPole-v1`. It compares DAgger with
`beta_i = 1[i=1]`, SMILe with `alpha=0.1`, and supervised expert-only data
collection for 20 iterations and five training seeds. Each method receives 500
new expert labels per iteration, and each checkpoint is evaluated on 20 fixed
episodes without expert help.

At the paper's checkpoints, all methods begin at failure rate 1.00 and mean
return 79.55. By iteration 5, supervised reaches return 500, DAgger reaches
496.65 with failure rate 0.03, and SMILe remains at failure rate 1.00 with
return 85.57. DAgger and supervised both measure 500 with no failures from
iteration 10 onward; SMILe reaches return 119.55 at iteration 20 but still
fails every evaluation episode before the 500-step limit. This is an adapted
CartPole result, not an experiment reported by the original paper.
