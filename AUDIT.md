# AUDIT — the Part 1 reproduction, read before trusting its numbers

Audit of the code now preserved unmodified under `legacy/`, performed at the start
of the compression/robustness extension. Nothing in `legacy/` was changed; every
correction lives in the new `src/` package.

The reproduction contains **two independent implementations** of LeNet-5, and they
disagree with each other on several points. That distinction matters throughout,
so they are named here:

| | path | what it is |
| --- | --- | --- |
| **Scripted path** | `legacy/src/` + `legacy/scripts/` | the CLI training pipeline; produced `legacy/logs/*` and the headline ~99.01% |
| **Notebook path** | `legacy/notebooks/experiment.ipynb` | a more literal reading of the 1998 paper; produced `legacy/notebooks/logs/lenet_paper/` |

---

## 1. Test-set leakage in best-epoch selection

**Confirmed, in the scripted path. This is the serious one.**

`legacy/src/data/loader.py:47` returns only a train and a test loader — no
validation set is constructed anywhere in the repo. `legacy/src/train.py:63-87`
then evaluates on the test set every epoch and keeps the checkpoint that maximises
*test* accuracy:

```python
accuracy = 100. * correct / len(test_loader.dataset)   # test set
...
if accuracy > best_acc:                                 # selection on test
    best_acc = accuracy
    torch.save(model.state_dict(), os.path.join(log_dir, 'best_model.pth'))
```

`best_acc` is what gets printed as the final result. So the reported ~99.01% is a
**maximum over 20 epochs of test accuracy**, not a single held-out measurement.
That is an optimistic estimator: over N epochs it selects the epoch whose test
noise happened to be most favourable. The bias is small on MNIST but it is a real
bias, it is in the same direction for every configuration, and it is comparable in
size to the differences the original write-up draws conclusions from.

The leakage is visible directly in the committed logs.
`legacy/logs/baseline_paper/training_log.csv` — the run the headline came from —
ends:

| epoch | 7 | 8 | **9** | 10 |
| --- | --- | --- | --- | --- |
| test acc | 98.79 | 98.93 | **99.01** | 98.94 |

The model that finished training scored **98.94%**. The reported **99.01%** is
epoch 9, kept because that epoch's test score happened to be the maximum. On this
run the bias is +0.07 pp — small, but it is measured on the same data it was
selected on, and the same +ε applies to every configuration in the write-up.

The notebook path avoids checkpoint selection (it saves the final epoch,
`experiment.ipynb` cell 4) but names the test loader `val_loss`/`val_acc`
throughout, which is the same conflation wearing a different label.

**Correction.** `src/data.py` carves a fixed 55,000 / 5,000 / 10,000 split with
`SPLIT_SEED = 1234`, held constant across training seeds. `src/train.py` selects
on validation only. It also records the leaky number alongside, purely so Phase 0
can report how large the bias was — see `results/tables/phase0_baseline.md`.

## 2. Seeding

**Scripted path: no seed at all.** There is no `torch.manual_seed`,
`np.random.seed` or `random.seed` anywhere under `legacy/src/` or
`legacy/scripts/`. Weight init, shuffling and augmentation sampling were all
unseeded, so none of `legacy/logs/*` is reproducible, and every reported figure is
a single unreproducible sample.

Worse, `legacy/src/data/loader.py:40` calls `np.random.shuffle(indices)` unseeded
to build the training-size subsets, so the `lenet_10pct` and `lenet_50pct` runs
used an unknown subset of the training data.

**Notebook path: a single fixed seed,** `SEED = 22` (cell 0), setting `random`,
`numpy` and `torch`. Reproducible, but n=1 — no variance estimate.

**Correction.** `src/utils.py:set_seed` seeds all three RNGs plus cuDNN
determinism; the dataloader shuffle uses an explicit seeded `torch.Generator`.
Everything runs over 5 seeds with mean ± std.

## 3. The C3 sparse connection table

**The presentation's claim is wrong in one direction and right in the other.**

- The **notebook path implements the real table.** `experiment.ipynb` cell 2
  defines `C3Layer` with all 16 connection lists from Table I of the paper
  (`[0,1,2]`, `[1,2,3]`, … `[0,1,2,3,4,5]`), built as 16 separate convolutions over
  channel subsets. It also implements `Subsampling` with the learnable per-channel
  scale, bias and sigmoid, `ScaledTanh` (1.7159·tanh(2x/3)), and an `RBFOutput`
  layer — a genuinely faithful reading of the paper.
- The **scripted path does not.** `legacy/src/models/lenet.py:19` is a plain
  `nn.Conv2d(6, 16, kernel_size=5)`: dense connectivity, all 6 input maps feeding
  all 16 output maps. It also uses plain `AvgPool2d` without the learnable
  scale/bias/sigmoid, and a plain `Linear(84, 10)` instead of the RBF output.

Since the ~99.01% headline came from the scripted path, **the headline number is
from the dense-C3 variant**, and the presentation's statement is accurate *for the
number actually reported*.

**Decision for Part 2.** `src/models.py` keeps dense C3, matching the scripted
path — it is the configuration the baseline number belongs to, and it is what the
pruning literature (Frankle & Carbin) uses. Sparse C3 is a hand-designed
connectivity prior, which would confound the question of which connections
*magnitude pruning* discovers. Worth revisiting as a Phase 3 ablation.

## 4. Input pipeline and augmentation

The two paths disagree on how 28×28 becomes 32×32:

| | scripted | notebook |
| --- | --- | --- |
| resize to 32×32 | `transforms.Resize((32,32))` — bilinear interpolation | `transforms.Pad(2)` — zero border |
| normalise | `Normalize((0.1307,), (0.3081,))` | same |

`Resize` resamples every pixel; `Pad` leaves the digit untouched. The paper pads.
This is not a large accuracy effect but it means the two paths were not training on
the same images.

Augmentation, when `--augment` was passed (`legacy/src/data/loader.py:18-23`), was
exactly two transforms, applied to the PIL image before tensor conversion:

```python
transforms.RandomRotation(10)                                # ±10°
transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))     # ±10% shift
```

No scaling, no shear, no elastic distortion — so this is **not** the affine
distortion set from §II.B of the paper, which the write-up implies. Test data was
never augmented (correct).

**Correction.** `src/models.py:InputNormalize` uses `Pad(2)` and moves both the
padding and the normalisation *inside* the model, so the network consumes raw
[0,1] pixels. `src/data.py` keeps the same two augmentation transforms so the
comparison against the original numbers stays meaningful, but they are now
documented as a subset of the paper's, not a reproduction of it.

## 5. Normalisation placement — why it is moved

Not a bug in Part 1, but a blocker for Part 2. With normalisation in the
dataloader, the tensor an attacker perturbs lives in standardised space, where an
L∞ ball of radius ε corresponds to a pixel-space ball of radius ε·0.3081. Running
PGD at "ε = 0.3" would then really be attacking at ε ≈ 0.092 in pixel space —
roughly a third of the intended budget — and would silently produce inflated
robust accuracy that looks like a defence.

`scripts/check_port.py` asserts the move is numerically exact: identical weights,
raw input to the new model vs pre-normalised input to the legacy model, max
absolute logit difference 0.00e+00, and identical parameter counts (61,706).

## 6. Other observations

- **Compute budget was consistent** — all seven logged runs used 10 epochs. Credit
  where due; this is the one methodology rule Part 1 already satisfied.
- **`legacy/logs/reproduction/` has a checkpoint but no `training_log.csv`**, so
  that run's curve cannot be recovered.
- **The MLP comparison is not parameter-matched.** `legacy/src/models/mlp.py` has
  ~296k parameters against LeNet-5's 61,706 — nearly 5×. "CNN beats MLP" from
  those two runs is confounded by the MLP being the larger model, so the
  comparison isolates nothing.
- **No dependency pinning.** `legacy/requirements.txt` lists bare package names
  with no versions; the environment that produced the numbers is unrecoverable.
- **No per-run metadata.** No git SHA, config, timing or hardware was recorded, so
  the logs cannot be attributed to a code state.

## 7. What this means for the reported results

Every Part 1 number is a single unseeded sample selected on the test set. The
differences the write-up draws conclusions from — notably the 99.01 vs 99.12
augmentation gap — are of the same order as the seed-to-seed variance of this
architecture, which nothing in Part 1 measured. Phase 0 re-runs the baseline over
5 seeds under both protocols to put a number on both effects; see
`results/tables/phase0_baseline.md`.

None of this makes the reproduction wrong about its main claim — LeNet-5 does
reach ~99% on MNIST. It makes the *second-order* claims unsupported.
