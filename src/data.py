"""MNIST data with a deterministic 55,000 / 5,000 / 10,000 split.

Two properties the legacy loader did not have:

1. A validation set exists at all. ``legacy/src/train.py`` selected its best
   epoch on the test set; there was nothing else to select on.
2. The train/val split is fixed by ``SPLIT_SEED``, independent of the training
   seed, so all five seeds see exactly the same validation images and the
   comparison across seeds is apples-to-apples.

Tensors come out in raw ``[0, 1]`` pixel space at 28x28. Padding and
normalisation happen inside the model (see ``src/models.py``).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms

SPLIT_SEED = 1234
N_TRAIN = 55_000
N_VAL = 5_000

# The 15 corruptions of Mu & Gilmer (2019), 10,000 test images each.
MNIST_C_CORRUPTIONS = [
    "brightness",
    "canny_edges",
    "dotted_line",
    "fog",
    "glass_blur",
    "impulse_noise",
    "motion_blur",
    "rotate",
    "scale",
    "shear",
    "shot_noise",
    "spatter",
    "stripe",
    "translate",
    "zigzag",
]

# Matches the augmentation in legacy/src/data/loader.py so the comparison
# against the original reported numbers stays meaningful.
AUGMENT_TRANSFORMS = [
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
]


@dataclass
class Loaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader


def _base_transform():
    return transforms.ToTensor()  # PIL uint8 -> float in [0, 1]


def train_val_indices(n_total: int = 60_000):
    """Fixed permutation split. Same for every training seed, by design."""
    g = torch.Generator().manual_seed(SPLIT_SEED)
    perm = torch.randperm(n_total, generator=g).tolist()
    return perm[:N_TRAIN], perm[N_TRAIN : N_TRAIN + N_VAL]


def get_loaders(
    batch_size: int = 64,
    eval_batch_size: int = 1000,
    data_dir: str = "./data",
    augment: bool = False,
    seed: int = 0,
    num_workers: int = 0,
    smoke: bool = False,
) -> Loaders:
    base = _base_transform()
    train_tf = transforms.Compose(AUGMENT_TRANSFORMS + [base]) if augment else base

    train_full = datasets.MNIST(data_dir, train=True, download=True, transform=train_tf)
    val_full = datasets.MNIST(data_dir, train=True, download=True, transform=base)
    test_set = datasets.MNIST(data_dir, train=False, download=True, transform=base)

    train_idx, val_idx = train_val_indices(len(train_full))
    if smoke:
        train_idx, val_idx = train_idx[:500], val_idx[:500]
        test_set = Subset(test_set, range(500))

    # Shuffling is seeded so a run is reproducible from its seed alone.
    g = torch.Generator().manual_seed(seed)
    return Loaders(
        train=DataLoader(
            Subset(train_full, train_idx),
            batch_size=batch_size,
            shuffle=True,
            generator=g,
            num_workers=num_workers,
        ),
        val=DataLoader(Subset(val_full, val_idx), batch_size=eval_batch_size, shuffle=False),
        test=DataLoader(test_set, batch_size=eval_batch_size, shuffle=False),
    )


def get_mnist_c_loaders(
    root: str = "./data/MNIST-C",
    corruptions=None,
    eval_batch_size: int = 1000,
    limit: int = None,
) -> dict:
    """One loader per corruption, in the same raw [0,1] space as the clean test set.

    Expects the Zenodo layout: ``<root>/<corruption>/test_images.npy`` of shape
    (10000, 28, 28, 1) uint8, plus ``test_labels.npy``. Fetch with
    ``scripts/download_mnistc.py``.
    """
    base = Path(root)
    if not base.exists():
        raise FileNotFoundError(
            f"MNIST-C not found at {base}. Run: python scripts/download_mnistc.py"
        )
    # Tolerate the archive being extracted one level deeper than expected.
    if not (base / "identity").exists() and (base / "mnist_c").exists():
        base = base / "mnist_c"

    wanted = corruptions or MNIST_C_CORRUPTIONS
    loaders = {}
    for name in wanted:
        d = base / name
        if not d.exists():
            raise FileNotFoundError(f"missing corruption directory: {d}")
        images = np.load(d / "test_images.npy")
        labels = np.load(d / "test_labels.npy")
        if limit:
            images, labels = images[:limit], labels[:limit]

        # (N, 28, 28, 1) uint8 -> (N, 1, 28, 28) float in [0,1], matching get_loaders.
        x = torch.from_numpy(images).permute(0, 3, 1, 2).float().div_(255.0)
        y = torch.from_numpy(labels).long()
        if x.shape[1:] != (1, 28, 28):
            raise ValueError(f"{name}: unexpected image shape {tuple(x.shape)}")
        loaders[name] = DataLoader(TensorDataset(x, y), batch_size=eval_batch_size, shuffle=False)
    return loaders
