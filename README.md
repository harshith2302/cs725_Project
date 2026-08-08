# Compressing and Stress-Testing LeNet-5

> **Status:** Phase 0 complete. Phases 1–5 in progress on `feat/compression-robustness`.
> This README is expanded as phases land; the full write-up is Phase 5.

Two parts in one repository.

**Part 1 — Reproduction.** LeNet-5 (LeCun et al., 1998) on MNIST, ~99% test accuracy,
with side experiments on an RBF output layer, data augmentation, training-set size,
and an MLP comparison. Preserved unmodified in [`legacy/`](legacy/), tagged
`v1-reproduction`.

**Part 2 — Extension.** The question MNIST accuracy alone cannot answer:

> Compression that is free on clean MNIST — is it free under distribution shift and
> adversarial attack?

Iterative magnitude pruning removes ~96% of LeNet's weights with no clean-accuracy
loss (Frankle & Carbin, 2019). What that does to MNIST-C corruption accuracy and
PGD-L∞ robust accuracy for this architecture has not been carefully checked. That
gap is the contribution.

The deliverable is a table with six columns per configuration — parameter count,
INT8 size, clean accuracy, MNIST-C mean accuracy, PGD robust accuracy, CPU latency —
every number mean ± std over 5 seeds.

## Phase 0 results — the baseline is not what was reported

Part 1 selected its best checkpoint on the **test set**. The reported 99.01% is epoch
9 of `legacy/logs/baseline_paper/training_log.csv`; the model that finished training
scored 98.94%. No seed was set anywhere in the scripted training path.

Re-run over 5 seeds with a proper 55,000 / 5,000 / 10,000 split and selection on
validation only:

| Budget | Protocol | standard | augmented | diff | p (Welch) |
|---|---|---|---|---|---|
| 10 ep | best epoch on validation | 98.88 ± 0.05 | 99.08 ± 0.09 | +0.20 | 0.003 |
| 10 ep | best epoch on test *(leaky)* | 98.93 ± 0.06 | 99.10 ± 0.07 | +0.17 | 0.004 |
| 20 ep | best epoch on validation | 99.08 ± 0.09 | 99.16 ± 0.04 | +0.08 | **0.115** |
| 20 ep | best epoch on test *(leaky)* | 99.13 ± 0.07 | 99.24 ± 0.03 | +0.11 | 0.023 |

Three findings:

1. **Test-set leakage is worth +0.05 ± 0.04 pp**, bracketing the +0.07 pp visible
   directly in the legacy log. Budget-matched, the honest baseline is 98.88 ± 0.05
   against the reported 99.01.
2. **The augmentation gain is budget-dependent.** Significant at the 10 epochs the
   original runs used (+0.20 pp, p=0.003), but not at 20 epochs under honest
   selection (+0.08 pp, p=0.115). Augmentation appears to buy convergence speed
   rather than final accuracy — the standard model catches up given twice the
   budget. The original conclusion was right; the single-seed evidence for it was
   not.
3. **Selection on test manufactures significance.** At 20 epochs the leaky protocol
   reports p=0.023 where the honest one reports p=0.115.

Full audit in [`AUDIT.md`](AUDIT.md); generated table in
[`results/tables/phase0_baseline.md`](results/tables/phase0_baseline.md).

## Reproducing

```bash
pip install -r requirements.txt
python scripts/check_port.py          # verifies the Part 1 -> Part 2 port
python scripts/run_phase0.py --smoke  # 1 epoch / 500 images, seconds
python scripts/run_phase0.py          # 5 seeds x 2 conditions, ~3.5 h on CPU
python scripts/make_phase0_table.py   # rebuild tables from results/raw/*.json
```

Every reported number traces to a JSON record in `results/raw/`, carrying its config
hash, seed, git SHA, machine spec and wall-clock time.

## Methodology

Rules enforced throughout, in [`METHODOLOGY.md`](METHODOLOGY.md) and `PROJECT_PLAN.md`: no
test-set selection, 5 seeds minimum with mean ± std, negative results reported as
found, fixed compute budget per condition, and normalisation as a layer *inside* the
model so PGD at ε=0.3 attacks raw pixels and means what the literature means.

## Layout

```
legacy/          Part 1, frozen and unmodified
src/             models.py  data.py  train.py  utils.py
scripts/         run_phase0.py  check_port.py  make_phase0_table.py
results/raw/     one JSON per run — the evidence behind every number
results/tables/  generated markdown, never hand-edited
```

## References

LeCun et al. (1998) *Gradient-Based Learning Applied to Document Recognition*, Proc. IEEE ·
Frankle & Carbin (2019) *The Lottery Ticket Hypothesis*, ICLR ·
Madry et al. (2018) *Towards Deep Learning Models Resistant to Adversarial Attacks*, ICLR ·
Mu & Gilmer (2019) *MNIST-C: A Robustness Benchmark for Computer Vision* ·
Hinton et al. (2015) *Distilling the Knowledge in a Neural Network*
