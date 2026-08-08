"""Correctness checks for the MNIST-C loader, against a Phase 0 checkpoint.

The decisive check is `identity`: MNIST-C ships an uncorrupted copy of the test
set, so accuracy on it must equal accuracy on the torchvision test set to within
rounding. That single equality validates the whole path at once -- the NHWC->NCHW
permutation, the uint8 /255 scaling, and the label alignment. A loader that got
any of them wrong would still produce plausible-looking corruption numbers.

    python scripts/check_mnistc.py
"""

import glob
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from src.data import MNIST_C_CORRUPTIONS, get_loaders, get_mnist_c_loaders
from src.evaluate import clean_accuracy, mnist_c_accuracy
from src.models import build_model


def main():
    matches = sorted(glob.glob(str(ROOT / "results/checkpoints/phase0-standard*seed0*.pth")))
    if not matches:
        sys.exit("no Phase 0 checkpoint found; run scripts/run_phase0.py first")
    model = build_model("lenet5")
    model.load_state_dict(torch.load(matches[0], map_location="cpu"))
    model.eval()
    device = torch.device("cpu")
    print(f"checkpoint: {Path(matches[0]).name}\n")

    clean = clean_accuracy(model, get_loaders(eval_batch_size=1000).test, device)
    identity = clean_accuracy(model, get_mnist_c_loaders(corruptions=["identity"])["identity"], device)
    print(f"clean (torchvision) : {clean:.2f}%")
    print(f"MNIST-C identity    : {identity:.2f}%")
    assert abs(clean - identity) < 0.05, (
        f"identity ({identity:.2f}%) != clean ({clean:.2f}%) -- the loader is misreading "
        "MNIST-C: check the NHWC->NCHW permutation, the /255 scaling and label alignment"
    )
    print("[ok] identity matches clean -- loader path verified end to end\n")

    loaders = get_mnist_c_loaders()
    assert len(loaders) == 15, f"expected 15 corruptions, got {len(loaders)}"
    print(f"[ok] all 15 corruptions present: {', '.join(MNIST_C_CORRUPTIONS[:4])}, ...")

    x, y = next(iter(loaders["fog"]))
    assert x.shape[1:] == (1, 28, 28), f"bad shape {tuple(x.shape)}"
    assert 0.0 <= x.min() <= x.max() <= 1.0, f"outside [0,1]: [{x.min():.3f}, {x.max():.3f}]"
    assert set(y.tolist()) <= set(range(10)), "labels outside 0-9"
    print(f"[ok] tensors are (N,1,28,28) in [{x.min():.2f}, {x.max():.2f}], labels 0-9")

    r = mnist_c_accuracy(model, loaders, device)
    print(f"\nmean over 15 corruptions: {r['mean_acc']:.2f}%")
    for name, acc in sorted(r["per_corruption"].items(), key=lambda kv: kv[1]):
        print(f"  {name:15s} {acc:6.2f}")

    # PROJECT_PLAN.md Phase 1.1: a clean-trained LeNet must score clearly below
    # 99% on the mean. If it does not, the loader is feeding it clean images.
    assert r["mean_acc"] < 95.0, (
        f"corruption mean {r['mean_acc']:.2f}% is implausibly high -- the loader is "
        "probably serving clean or mislabelled data"
    )
    print(f"\n[ok] corruption mean {r['mean_acc']:.2f}% is well below clean {clean:.2f}%")
    print("\nall MNIST-C checks passed")


if __name__ == "__main__":
    main()
