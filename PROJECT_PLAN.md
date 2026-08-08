# PROJECT PLAN — Compressing and Stress-Testing LeNet-5

> **How to use this file.** It lives at the repo root and is the spec. Execute one
> phase at a time, reviewing the diff before each commit. Running several phases in
> one go loses the ability to catch methodological errors.

---

## 1. Context

This repo currently contains a reproduction of LeNet-5 (LeCun et al., 1998) on MNIST,
achieving ~99.01% test accuracy, plus side experiments (RBF output layer, data
augmentation, training-set-size sweep, MLP comparison). That work is complete and
should be **preserved, not overwritten**.

This project extends it along two axes that MNIST accuracy alone cannot measure:
**compression** and **robustness**.

### Research question

> Compression that is free on clean MNIST — is it free under distribution shift and
> adversarial attack?

Iterative magnitude pruning can remove ~96% of LeNet's weights with no clean-accuracy
loss (Frankle & Carbin, ICLR 2019). Nobody has carefully checked what that does to
MNIST-C corruption accuracy or PGD robust accuracy for this architecture. That gap is
the contribution.

### Deliverable

A results table with **six columns per configuration**: parameter count, INT8 model
size, clean accuracy, MNIST-C mean accuracy, PGD-L∞ (ε=0.3) robust accuracy, CPU
latency. Every number is mean ± std over 5 seeds.

---

## 2. Non-negotiable methodology rules

These are the difference between a course project and credible work. The agent must
enforce them and refuse to produce results that violate them.

1. **No test-set model selection.** Carve a 5,000-image validation set out of the
   60,000 training images with a fixed seed. Select best epoch / early stopping /
   any hyperparameter on validation only. Touch the test set once, at the end, per
   configuration. The existing repo likely selects best-epoch on test — fix this and
   document it as a correction.
2. **5 seeds minimum, report mean ± std.** A single-seed difference of 0.1% on MNIST
   is noise. Any claimed effect must exceed seed variance.
3. **Report the negative result.** If pruning does not hurt robustness, that is the
   finding. Do not tune until the hypothesis is confirmed.
4. **Attacks operate in [0,1] pixel space.** The current pipeline normalises with
   mean=0.1307, std=0.3081. Move normalisation *inside* the model as its first layer
   so that PGD perturbs raw pixels with ε=0.3 meaning what the literature means.
   Getting this wrong silently invalidates every adversarial number.
5. **Log everything.** Every run writes a JSON record: config hash, seed, git commit
   SHA, all metrics, wall-clock time. No metric appears in a figure unless it came
   from a JSON file on disk.
6. **Fixed compute budget per condition.** When comparing dense vs pruned, give both
   the same number of training epochs. Do not let one condition train longer.

---

## 3. Target repo structure

Restructure the repo into this layout. Move existing reproduction code into
`legacy/` untouched, and tag the current HEAD as `v1-reproduction` before anything.

```
.
├── PROJECT_PLAN.md
├── README.md                  # rewritten at the end
├── requirements.txt
├── legacy/                    # existing reproduction code, unmodified
├── src/
│   ├── __init__.py
│   ├── models.py              # LeNet5 with normalisation as layer 0, MLP, wide teacher
│   ├── data.py                # train/val/test splits, augmentation, MNIST-C loader
│   ├── train.py               # single training run, seed-controlled
│   ├── prune.py               # iterative magnitude pruning + rewinding
│   ├── distill.py             # Hinton KD
│   ├── quantize.py            # QAT + INT8 export
│   ├── attacks.py             # PGD, FGSM
│   ├── evaluate.py            # clean / MNIST-C / adversarial / latency
│   └── utils.py               # seeding, logging, checkpointing, param counting
├── configs/                   # one YAML per experiment condition
├── scripts/
│   ├── download_mnistc.sh
│   ├── run_all.sh
│   ├── kaggle_push.py         # push + poll + pull results via Kaggle API
│   └── make_figures.py
├── results/
│   ├── raw/                   # one JSON per run
│   ├── tables/                # generated CSV/markdown
│   └── figures/
├── notebooks/
│   └── colab_runner.ipynb     # clone repo, run configs, commit results back
└── report/
    └── report.md              # final writeup
```

**Git workflow:** all work on branch `feat/compression-robustness`. One commit per
logical unit. Never force-push. Tag `v1-reproduction` on the current main first.

---

## 4. Phases

### Phase 0 — Audit and fix the existing code

**Goal:** establish a trustworthy baseline before adding anything.

1. Tag current HEAD: `git tag v1-reproduction && git push --tags`.
2. Create branch `feat/compression-robustness`.
3. Move existing code to `legacy/`. Do not edit it.
4. Read the legacy training loop. Write `AUDIT.md` documenting:
   - whether best-epoch selection used the test set (test leakage)
   - whether any seed was fixed
   - whether the C3 sparse connection table from the original paper was implemented
     (the presentation says it was not — dense connections were used instead)
   - the exact augmentation transforms used
5. Re-implement `src/models.py` with LeNet-5 where **normalisation is a module inside
   the network**, so the model accepts raw [0,1] tensors. Verify it reproduces
   ~99% to confirm the port is faithful.
6. Implement `src/data.py` with a deterministic 55,000 / 5,000 / 10,000 split.
7. Re-run the baseline with 5 seeds. Report mean ± std for both the leaky
   (best-epoch-on-test) and correct (best-epoch-on-val) protocols.

**Expected output:** a table showing the leakage-corrected baseline is slightly lower
than 99.01%, and that the 99.01 vs 99.12 augmentation gap from the original slides
falls inside seed variance. This is a real finding — put it in the report.

**Compute:** CPU is fine. ~15 min per run.

---

### Phase 1 — Evaluation suite

Build all measurement code before building any method code. Everything downstream
depends on these being correct.

1. **MNIST-C loader.**
   Download from Zenodo record 3239543 (`mnist_c.zip`, ~15 corruption directories:
   shot noise, impulse noise, glass blur, motion blur, shear, scale, rotate,
   brightness, translate, stripe, fog, spatter, dotted line, zigzag, canny edges;
   10,000 test images each). Source code: `github.com/google-research/mnist-c`.
   Report **per-corruption accuracy and the mean across all 15** — the mean alone
   hides which corruptions break which models.
   Sanity check: a clean-trained LeNet should score clearly below 99% on the mean;
   if it does not, the loader is wrong.

2. **PGD attack** (`src/attacks.py`).
   L∞, ε=0.3, step size α=0.01, 40 steps, random start inside the ε-ball, clamp to
   [0,1] after every step. This is the Madry et al. (2018) MNIST threat model.
   Sanity check: a standard clean-trained LeNet-5 should collapse to roughly
   10–20% accuracy under this attack. Published work reports ~17.6% for LeNet-5 at
   ε=0.3. If you get 80%, gradient masking or a normalisation bug is present —
   stop and debug, do not report it.
   Also implement FGSM as a cheap sanity attack and an ε-sweep
   (0.05 → 0.4) for the robustness curve figure.

3. **Latency benchmark.**
   CPU, batch size 1, single thread, 100 warm-up iterations then 1000 timed;
   report the **median**, not the mean. Also report throughput at batch 128.
   Record the machine spec in the JSON — latency numbers are meaningless without it.

4. **Model size accounting.**
   Total params, non-zero params, FP32 checkpoint size on disk, INT8 size on disk.
   Note explicitly that unstructured sparsity does **not** shrink a dense checkpoint —
   report both the theoretical sparse size (CSR) and the actual dense file size.
   Being honest about this is a credibility signal.

**Compute:** CPU. MNIST-C eval ~2 min; PGD-40 on 10k images ~3 min on GPU.

---

### Phase 2 — Compression pipeline

1. **Iterative magnitude pruning (`src/prune.py`).**
   Follow Frankle & Carbin: train → prune the lowest-magnitude 20% of weights
   layer-wise → **reset surviving weights to their initial values** → retrain.
   Prune the output layer at half rate. Repeat for ~20 rounds to reach ~1.2%
   remaining weights.
   Implement three variants for comparison:
   - **LTH rewind-to-init** (the original)
   - **Late rewinding** to epoch k (Frankle et al. 2020) — often more stable
   - **Fine-tuning** without reset (the Han et al. 2015 baseline)
   Save a checkpoint at each sparsity level. Key levels to carry forward:
   100%, 51.2%, 21.1%, 13.5%, 3.6%, 1.2% weights remaining.
   Sanity check: accuracy should hold flat or improve slightly down to ~13%, then
   degrade. If it degrades immediately, the mask is being applied incorrectly.

2. **Knowledge distillation (`src/distill.py`).**
   Teacher: a wider LeNet (2× channels) or the dense baseline.
   Student: the pruned network, or a narrow LeNet.
   Standard Hinton KD: `L = α·T²·KL(student/T ‖ teacher/T) + (1-α)·CE`, T=4, α=0.7.
   Ablate T ∈ {1, 4, 10} and α ∈ {0.3, 0.7, 0.9} on validation only.

3. **Quantization-aware training (`src/quantize.py`).**
   `torch.ao.quantization`, QAT with fake-quant observers, backend `qnnpack` (ARM)
   or `fbgemm` (x86). Fuse Conv+Act before quantising. Export INT8 and measure
   real on-disk size and real CPU latency.
   **Important limitation to document:** a true INT8 model has no gradients, so PGD
   cannot attack it directly. Evaluate adversarial robustness on the fake-quantised
   FP32 graph (the QAT model in eval mode) and state clearly in the report that this
   is a surrogate. Do not silently attack the FP32 model and label it INT8.

4. **Adversarial training (Madry).**
   PGD-40 adversarial training at ε=0.3 as a separate training mode. This becomes a
   third training condition alongside standard and augmented.

**Compute:** IMP is the expensive part — 20 rounds × 20 epochs × 5 seeds × 3 variants.
Estimate ~8–12 GPU hours total. This is the phase to run on Kaggle.

---

### Phase 3 — The experiment grid

Do not run the full cross-product; it explodes. Run this:

**Grid A — sparsity sweep (the main result).**
`{standard, augmented, adversarially-trained} × {100%, 21.1%, 13.5%, 3.6%, 1.2%} × 5 seeds`
Evaluate all six metrics on each. 75 runs.

**Grid B — full pipeline (the deployment result).**
`{dense, IMP@3.6%, IMP@3.6% + KD, IMP@3.6% + KD + QAT} × 5 seeds`
20 runs.

**Grid C — ablations (cheap, single sparsity level).**
- rewind-to-init vs late-rewinding vs fine-tuning
- KD temperature/alpha sweep
- ε-sweep robustness curves for dense vs pruned

**The plot that carries the paper:** x-axis = fraction of weights remaining (log
scale), three lines = clean accuracy, MNIST-C mean accuracy, PGD robust accuracy,
with shaded ±1 std bands. If the three curves diverge — clean stays flat while
robustness falls off earlier — that single figure is the entire contribution.

---

### Phase 4 — Analysis

1. **Statistical testing.** For every claimed difference, run Welch's t-test across
   seeds and report the p-value. Explicitly state which differences are *not*
   significant.
2. **Per-corruption breakdown.** Heatmap: sparsity level × 15 corruption types. Look
   for whether specific corruptions (e.g. blur, fog) degrade faster than others —
   that suggests which filters pruning removes first.
3. **Filter visualisation.** Show which C1 filters survive at 3.6% sparsity. Are they
   the oriented edge detectors? This connects directly back to LeCun's original
   argument about learned feature hierarchies.
4. **Replicate the MNIST-C paper's counterintuitive claim.** Mu & Gilmer report that
   several adversarial defences *degrade* corruption robustness. Test it: does your
   adversarially-trained LeNet score worse on MNIST-C than the standard one?

---

### Phase 5 — Writeup and packaging

1. `report/report.md` — 6–8 pages: motivation, methodology corrections from Phase 0,
   setup, results tables, the three-curve figure, discussion, limitations,
   references.
2. Rewrite `README.md`: what the repo contains, how to reproduce every number, the
   headline results table, and a clear "Part 1: reproduction / Part 2: extension"
   split.
3. `results/tables/main.md` — the six-column table, auto-generated from JSON.
4. Open a PR from `feat/compression-robustness` to `main` with a written summary.
   A well-written PR description is itself a portfolio artifact.
5. Draft 3 resume bullets with real numbers filled in.

---

## 5. Compute orchestration

### Local (development)
Everything except Phase 2 runs fine on CPU. Use a `--smoke` flag on every script
(1 epoch, 500 images) so the agent can verify correctness in seconds before
committing to a real run.

### Kaggle (the automatable path — use this for Phase 2 and 3)

Setup once:
```bash
pip install kaggle
# put kaggle.json (from kaggle.com → Settings → API → Create New Token)
# at ~/.kaggle/kaggle.json, then: chmod 600 ~/.kaggle/kaggle.json
```

The full loop then runs without touching a browser:
```bash
kaggle kernels push -p kaggle_kernel/     # submit
kaggle kernels status <user>/<kernel>     # poll until "complete"
kaggle kernels output <user>/<kernel> -p results/raw/   # pull artifacts
```

`scripts/kaggle_push.py` should: write `kernel-metadata.json` (set
`enable_gpu: true`, `enable_internet: true`), push, poll every 60s, pull outputs into
`results/raw/`, and commit them. The kernel script itself should `git clone` this repo
at a pinned commit SHA, run the requested configs, and write JSON to `/kaggle/working/`.

Free tier: ~30 GPU hours/week, 9h max per session, 20GB output. This project needs
well under that. Split Grid A into per-seed kernels so no single run exceeds 9h.

### Colab (fallback — manual trigger)
There is no supported API for headless Colab execution. The workflow is:
`notebooks/colab_runner.ipynb` clones the repo, installs deps, runs configs passed as
a parameter, and pushes results back to a `results-colab` branch using a GitHub PAT
stored in Colab Secrets. You open it manually via
`colab.research.google.com/github/<user>/<repo>/blob/main/notebooks/colab_runner.ipynb`.
Use this only if Kaggle quota runs out.

---

## 6. Environment

```
torch>=2.2
torchvision
numpy
pandas
matplotlib
seaborn
pyyaml
tqdm
scipy            # statistical tests
torchattacks     # optional — but implement PGD yourself first and verify against it
```

Pin versions in `requirements.txt`. Quantization backends differ between PyTorch
versions; record the exact version in every JSON log.

---

## 7. Guardrails for the agent

The agent must **stop and ask** rather than proceed if:

- A metric looks too good (e.g. >50% PGD robust accuracy on a clean-trained model).
  This almost always means a bug, not a discovery.
- Reproducing the baseline requires changing the protocol to hit 99.01%.
- Pruning results contradict Frankle & Carbin (accuracy collapsing immediately at
  low sparsity) — debug the masking before reporting.
- A run would exceed 2 hours locally without being sent to Kaggle.

The agent must **never**:

- Select checkpoints, early-stopping points, or hyperparameters using test data.
- Report a single-seed number as a result.
- Silently drop a seed or configuration whose result was inconvenient.
- Modify anything in `legacy/`.
- Claim a difference is meaningful without a seed-variance check.

---

## 8. References to cite in the report

- LeCun, Bottou, Bengio, Haffner (1998). *Gradient-Based Learning Applied to Document Recognition.* Proc. IEEE.
- Frankle & Carbin (2019). *The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks.* ICLR. arXiv:1803.03635
- Frankle, Dziugaite, Roy, Carbin (2020). *Linear Mode Connectivity and the Lottery Ticket Hypothesis.* ICML. (late rewinding)
- Han, Pool, Tran, Dally (2015). *Learning both Weights and Connections for Efficient Neural Networks.* NeurIPS.
- Hinton, Vinyals, Dean (2015). *Distilling the Knowledge in a Neural Network.* arXiv:1503.02531
- Jacob et al. (2018). *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference.* CVPR.
- Madry, Makelov, Schmidt, Tsipras, Vladu (2018). *Towards Deep Learning Models Resistant to Adversarial Attacks.* ICLR. arXiv:1706.06083
- Mu & Gilmer (2019). *MNIST-C: A Robustness Benchmark for Computer Vision.* arXiv:1906.02337
- Hendrycks & Dietterich (2019). *Benchmarking Neural Network Robustness to Common Corruptions and Perturbations.* ICLR.
- Tsipras, Santurkar, Engstrom, Turner, Madry (2019). *Robustness May Be at Odds with Accuracy.* ICLR.

---

## 9. Resume bullets (fill in after Phase 4)

> Compressed LeNet-5 to __% of original parameters via iterative magnitude pruning
> with rewinding, retaining __% clean MNIST accuracy; INT8 quantization-aware
> training reduced the deployed model to __ KB with __× CPU inference speedup.

> Benchmarked sparse and quantized CNNs against 15 MNIST-C corruption types and
> PGD-L∞ attacks (ε=0.3), showing compression that is free on clean accuracy costs
> __ percentage points of corruption robustness — an effect invisible to standard
> evaluation.

> Identified and corrected test-set leakage in a prior baseline; re-evaluated all
> results over 5 seeds with significance testing, demonstrating the previously
> reported augmentation gain fell within seed variance.
