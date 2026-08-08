"""Port-fidelity checks for the Phase 0 rewrite.

Confirms the three properties everything downstream relies on:

1. The new model has the same parameter count as the legacy one -- the port
   moved normalisation, it did not change the architecture.
2. Moving normalisation inside the network is numerically a no-op: feeding raw
   [0,1] pixels to the new model equals feeding pre-normalised tensors to the
   legacy model, given identical weights.
3. The train/val/test split is deterministic and disjoint.
"""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from src.data import N_TRAIN, N_VAL, train_val_indices
from src.models import MNIST_MEAN, MNIST_STD, LeNet5
from src.utils import count_params


def _load_legacy_lenet():
    """Load legacy/src/models/lenet.py by path.

    A plain import cannot work: ``src`` already names the new package, and the
    legacy file must stay unmodified (PROJECT_PLAN.md section 7).
    """
    spec = importlib.util.spec_from_file_location(
        "legacy_lenet", ROOT / "legacy" / "src" / "models" / "lenet.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.LeNet5


LegacyLeNet5 = _load_legacy_lenet()


def check_param_count():
    new, legacy = LeNet5(), LegacyLeNet5()
    n_new = count_params(new)["total_params"]
    n_legacy = sum(p.numel() for p in legacy.parameters())
    assert n_new == n_legacy, f"param mismatch: new {n_new} vs legacy {n_legacy}"
    print(f"[ok] parameter count identical: {n_new:,}")


def check_normalisation_equivalence():
    new, legacy = LeNet5(), LegacyLeNet5()
    legacy.load_state_dict({k: v for k, v in new.state_dict().items() if not k.startswith("normalize.")})
    new.eval(), legacy.eval()

    raw = torch.rand(8, 1, 28, 28)  # raw pixels in [0, 1], what the new model expects
    pre = (F.pad(raw, (2, 2, 2, 2)) - MNIST_MEAN) / MNIST_STD  # what the legacy loader produced

    with torch.no_grad():
        delta = (new(raw) - legacy(pre)).abs().max().item()
    assert delta < 1e-5, f"outputs diverge by {delta}"
    print(f"[ok] in-model normalisation is a no-op: max |diff| = {delta:.2e}")


def check_split():
    tr1, va1 = train_val_indices()
    tr2, va2 = train_val_indices()
    assert tr1 == tr2 and va1 == va2, "split is not deterministic across calls"
    assert len(tr1) == N_TRAIN and len(va1) == N_VAL
    assert not (set(tr1) & set(va1)), "train and val overlap"
    print(f"[ok] split deterministic and disjoint: {len(tr1):,} train / {len(va1):,} val")


def check_input_range():
    """PGD assumes the model's input domain is [0, 1]; assert nothing rescales it."""
    from src.data import get_loaders

    x, _ = next(iter(get_loaders(batch_size=64, smoke=True).test))
    assert 0.0 <= x.min() <= x.max() <= 1.0, f"inputs outside [0,1]: [{x.min()}, {x.max()}]"
    print(f"[ok] dataloader emits raw pixels in [{x.min():.3f}, {x.max():.3f}], shape {tuple(x.shape)}")


if __name__ == "__main__":
    torch.manual_seed(0)
    check_param_count()
    check_normalisation_equivalence()
    check_split()
    check_input_range()
    print("\nall port checks passed")
