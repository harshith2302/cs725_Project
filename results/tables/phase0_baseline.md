# Phase 0 — leakage-corrected baseline

LeNet-5 (dense C3, 61,706 params), sgd lr=0.01 momentum=0.9, batch 64, 20 epochs.
Split 55,000 train / 5,000 val / 10,000 test, `SPLIT_SEED=1234` fixed across seeds.
Machine: Windows-10-10.0.26200-SP0, torch 2.10.0+cpu, CUDA False.
Generated from `results/raw/*.json` at commit `838e9986110a0961d941691191c4483238b9c15c-dirty`.

## Test accuracy (%), mean ± std over seeds

| Protocol | standard (n=5) | augmented (n=5) |
| --- | --- | --- |
| Correct - best epoch on validation | 99.08 ± 0.09 | 99.16 ± 0.04 |
| Leaky - best epoch on test | 99.13 ± 0.07 | 99.24 ± 0.03 |
| Final epoch (no selection) | 99.08 ± 0.10 | 99.15 ± 0.06 |

## Size of the test-set leakage

Per-seed difference between the two selection protocols on the *same* run.

| Condition | Leaky − correct (pp) | per-seed |
| --- | --- | --- |
| standard | 0.05 ± 0.04 | +0.00, +0.05, +0.07, +0.10, +0.03 |
| augmented | 0.08 ± 0.05 | +0.10, +0.15, +0.04, +0.03, +0.08 |

## Per-seed detail

| Condition | Seed | Best val epoch | Val acc | Test @ best val | Leaky test |
| --- | --- | --- | --- | --- | --- |
| standard | 0 | 17 | 99.16 | 99.20 | 99.20 |
| standard | 1 | 19 | 99.12 | 99.15 | 99.20 |
| standard | 2 | 19 | 99.20 | 98.99 | 99.06 |
| standard | 3 | 19 | 99.06 | 99.04 | 99.14 |
| standard | 4 | 15 | 99.04 | 99.02 | 99.05 |
| augmented | 0 | 18 | 99.22 | 99.13 | 99.23 |
| augmented | 1 | 13 | 99.20 | 99.14 | 99.29 |
| augmented | 2 | 19 | 99.12 | 99.21 | 99.25 |
| augmented | 3 | 20 | 99.26 | 99.19 | 99.22 |
| augmented | 4 | 19 | 98.88 | 99.14 | 99.22 |

## Is the augmentation gain real?

Welch's t-test across the 5 seeds, two-sided. Both training budgets and both
selection protocols are reported: which cell you look at changes the answer,
which is itself the finding.

| Budget | Protocol | standard | augmented | diff (pp) | t | df | p | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10 ep | best val | 98.88 ± 0.05 | 99.08 ± 0.09 | +0.20 | 4.45 | 6.7 | 0.003 | sig. |
| 10 ep | best test (leaky) | 98.93 ± 0.06 | 99.10 ± 0.07 | +0.17 | 4.01 | 7.8 | 0.004 | sig. |
| 20 ep | best val | 99.08 ± 0.09 | 99.16 ± 0.04 | +0.08 | 1.89 | 5.2 | 0.115 | **not sig.** |
| 20 ep | best test (leaky) | 99.13 ± 0.07 | 99.24 ± 0.03 | +0.11 | 3.19 | 5.3 | 0.023 | sig. |

For reference, the original single-seed write-up reported 99.01 (standard) vs 99.12 (augmented), a +0.11 pp gap, and treated it as a real effect.

## Budget-matched against the legacy 10-epoch protocol

The legacy runs trained for 10 epochs; these for 20. Because the learning rate is
constant with no scheduler, the first 10 epochs of each run are identical to a
10-epoch run at the same seed, so the shorter budget is re-derived exactly from the
stored curves rather than retrained (`truncate_budget`).

| Condition | Budget | Correct (best val) | Leaky (best test) | Legacy reported |
| --- | --- | --- | --- | --- |
| standard | 10 epochs | 98.88 ± 0.05 | 98.93 ± 0.06 | 99.01 |
| standard | 20 epochs | 99.08 ± 0.09 | 99.13 ± 0.07 | — |
| augmented | 10 epochs | 99.08 ± 0.09 | 99.10 ± 0.07 | 99.12 |
| augmented | 20 epochs | 99.16 ± 0.04 | 99.24 ± 0.03 | — |

The legacy numbers were produced under the *leaky* 10-epoch protocol, so that is the
column they should be read against.

