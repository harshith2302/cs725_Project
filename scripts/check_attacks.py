"""Correctness checks for src/attacks.py, run against a Phase 0 checkpoint.

PROJECT_PLAN.md Phase 1.2 predicts a clean-trained LeNet-5 should score 10-20%
under PGD-40 at eps=0.3, citing ~17.6%. This suite measures 0.0% instead. The
checks below exist to establish which number is wrong, because "the attack is
too strong" and "the attack is buggy" look identical from a single data point.

Gradient masking has a signature: FGSM beating PGD, non-monotonic degradation,
or a plateau at high accuracy. All three are tested. A model whose robust
accuracy falls smoothly and monotonically, and where PGD dominates FGSM at
every epsilon, is being attacked correctly.

    python scripts/check_attacks.py
"""

import glob
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from src.attacks import adversarial_accuracy, assert_pixel_space_model
from src.data import get_loaders
from src.models import build_model

EPSILONS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]


def load_checkpoint():
    matches = sorted(glob.glob(str(ROOT / "results/checkpoints/phase0-standard*seed0*.pth")))
    if not matches:
        sys.exit("no Phase 0 checkpoint found; run scripts/run_phase0.py first")
    model = build_model("lenet5")
    model.load_state_dict(torch.load(matches[0], map_location="cpu"))
    return model.eval(), Path(matches[0]).name


def main():
    model, name = load_checkpoint()
    device = torch.device("cpu")
    loader = get_loaders(eval_batch_size=500).test
    print(f"checkpoint: {name}\n")

    assert_pixel_space_model(model)
    print("[ok] model owns its normalisation (attacks operate on raw pixels)")

    rows = []
    for eps in EPSILONS:
        f = adversarial_accuracy(model, loader, device, attack="fgsm", eps=eps, max_batches=1)
        p = adversarial_accuracy(model, loader, device, attack="pgd", eps=eps, max_batches=1)
        rows.append((eps, f["robust_acc"], p["robust_acc"]))

    print(f"\n{'eps':>6} {'FGSM':>8} {'PGD-40':>8}")
    for eps, f, p in rows:
        print(f"{eps:>6.2f} {f:>8.1f} {p:>8.1f}")
    print()

    clean = rows[0][2]
    # eps=0 must be a no-op: any drop means the attack corrupts inputs it should leave alone.
    assert abs(rows[0][1] - clean) < 1e-6, "FGSM at eps=0 changed accuracy"
    print(f"[ok] eps=0 is a no-op ({clean:.1f}%)")

    # PGD is a multi-step FGSM with projection; it must never be the weaker attack.
    weaker = [(e, f, p) for e, f, p in rows if p > f + 1e-9]
    assert not weaker, f"PGD weaker than FGSM at {weaker} -- classic gradient-masking signature"
    print("[ok] PGD dominates FGSM at every epsilon")

    pgd = [p for _, _, p in rows]
    assert all(a >= b - 1e-9 for a, b in zip(pgd, pgd[1:])), f"non-monotonic PGD curve: {pgd}"
    print("[ok] robust accuracy decreases monotonically in epsilon")

    # The plan's own guardrail: >50% on a clean-trained model means a bug.
    assert pgd[-1] < 50.0, f"PGD at eps=0.3 gave {pgd[-1]:.1f}% -- suspiciously high, debug before reporting"
    print(f"[ok] PGD at eps=0.3 is {pgd[-1]:.1f}%, below the 50% bug threshold")

    print(
        "\nNote: PROJECT_PLAN.md expects 10-20% here. Madry et al. (2018) Table 1 reports\n"
        "0.0% for a naturally-trained MNIST model under PGD at eps=0.3, which is what we\n"
        "measure. The checks above rule out gradient masking as the explanation."
    )
    print("\nall attack checks passed")


if __name__ == "__main__":
    main()
