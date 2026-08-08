# METHODOLOGY.md

Working rules for this repository.

## What this repo is

Two parts, one history.

- **Part 1 — reproduction.** LeNet-5 (LeCun et al., 1998) on MNIST, ~99% test
  accuracy, plus side experiments. Frozen in `legacy/`, tagged `v1-reproduction`.
- **Part 2 — extension.** Compression (iterative magnitude pruning, distillation,
  INT8 QAT) measured against corruption and adversarial robustness, not just clean
  accuracy. Everything outside `legacy/`.

The research question: *compression that is free on clean MNIST — is it free under
distribution shift and adversarial attack?*

`PROJECT_PLAN.md` is the spec. Work one phase at a time; do not run ahead into the
next phase without being asked.

## Non-negotiable methodology rules

These are the difference between a course project and credible work. Enforce them,
and refuse to produce results that violate them.

1. **No test-set model selection.** Select epochs, early stopping and
   hyperparameters on the 5,000-image validation split only. Touch the test set
   once per configuration, at the end. This was the central defect in Part 1 — see
   `AUDIT.md`.
2. **5 seeds minimum, report mean ± std.** Seed variance on this architecture is
   ~0.09 pp. A single-seed difference below that is noise. Never report a
   single-seed number as a result.
3. **Report the negative result.** If pruning doesn't hurt robustness, that is the
   finding. Never tune until a hypothesis is confirmed.
4. **Attacks operate in [0,1] pixel space.** Normalisation is a module inside the
   model (`src/models.py:InputNormalize`), never a dataset transform. If you ever
   move it back into the dataloader, every adversarial number silently becomes
   wrong by a factor of 1/0.3081.
5. **Every run writes a JSON record** to `results/raw/` with config hash, seed, git
   SHA, metrics and wall-clock. No metric appears in a table or figure unless it
   came from a JSON file on disk.
6. **Fixed compute budget per condition.** When comparing conditions, give them the
   same number of epochs. Phase 0 found the augmentation effect is budget-dependent
   — this rule is load-bearing, not bureaucratic.

## Hard constraints

- **Never modify anything in `legacy/`.** It is the preserved Part 1 artifact.
  `scripts/check_port.py` loads from it by file path via `importlib` precisely
  because a normal import would collide with the new `src` package.
- **Never commit credentials.** The Kaggle token belongs in `~/.kaggle/kaggle.json`,
  outside the repo. `kaggle.json`, `.kaggle/` and `*.env` are gitignored.
- **All work on `feat/compression-robustness`.** One commit per logical unit. Never
  force-push.

## Stop and ask rather than proceed if

- A metric looks too good — e.g. >50% PGD robust accuracy on a clean-trained model.
  That is a bug, not a discovery. Published work reports ~17.6% for LeNet-5 at
  ε=0.3; anything near 80% means gradient masking or a normalisation error.
- Reproducing a baseline requires changing the protocol to hit a target number.
- Pruning contradicts Frankle & Carbin — accuracy collapsing immediately at low
  sparsity means the mask is being applied wrongly. Debug before reporting.
- A run would exceed 2 hours locally without being sent to a GPU.

## Layout

```
legacy/          Part 1, frozen. Do not edit.
src/             models.py data.py train.py utils.py  (+ prune/distill/quantize/attacks/evaluate as phases land)
scripts/         run_phase0.py check_port.py make_phase0_table.py
results/raw/     one JSON per run — committed, this is the evidence
results/tables/  generated markdown — never hand-edited
results/checkpoints/, results/smoke/   gitignored
```

## Commands

```bash
python scripts/check_port.py                  # port-fidelity assertions
python scripts/run_phase0.py --smoke          # 1 epoch, 500 images, seconds
python scripts/run_phase0.py                  # 5 seeds x 2 conditions, ~3.5h CPU
python scripts/make_phase0_table.py           # rebuild tables from results/raw/*.json
```

Every training script takes `--smoke`. Use it before committing to a real run.

## Conventions that exist for a reason

- **Smoke runs never write to `results/raw/`.** `train_one` routes them to
  `results/smoke/`, and the table builder rejects any record with `config.smoke`
  set. A 1-epoch record averaged into a reported baseline is a silent, plausible
  corruption — hence two independent guards.
- **Tables are rebuilt from all JSONs on disk**, not from in-memory state, so
  running one condition cannot silently drop another from the table.
- **`truncate_budget()` in `make_phase0_table.py` is only valid without an LR
  schedule.** It re-derives a shorter budget from a stored epoch curve, which works
  because the first N epochs of a longer run are bit-identical at the same seed.
  Adding a scheduler invalidates it.
- **Windows console is cp1252** and cannot encode `±`/`−`/`α`. Scripts that print
  tables reconfigure stdout to UTF-8 at import; the files are always UTF-8.

## Environment

CPU-only PyTorch locally (~57 s/epoch for LeNet-5 on 55k images). Phase 2 needs a
GPU. `results/raw/*.json` records the exact torch version and machine spec per run,
because quantization backends differ between versions and latency numbers are
meaningless without hardware context.
