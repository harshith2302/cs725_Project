"""Models for the compression / robustness study.

The one structural change from ``legacy/src/models/lenet.py`` is that padding
and normalisation are **modules inside the network** rather than dataset
transforms. Every model here therefore accepts a raw ``(B, 1, 28, 28)`` tensor
with values in ``[0, 1]``.

That matters for Phase 1: PGD with epsilon=0.3 is defined on raw pixels. If
normalisation sits in the dataloader, an epsilon-ball in the tensor the attack
sees corresponds to an epsilon/0.3081 ball in pixel space, and every
adversarial number is silently wrong.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


class InputNormalize(nn.Module):
    """Pad 28x28 -> 32x32 and standardise, as layer 0 of the network.

    Zero-padding (rather than the ``Resize`` used in ``legacy/src/data/loader.py``)
    matches the original paper, and keeps the attack surface exactly the 28x28
    MNIST image: the padded border is a deterministic function of the input, not
    something an adversary can perturb.
    """

    def __init__(self, mean: float = MNIST_MEAN, std: float = MNIST_STD, pad: int = 2):
        super().__init__()
        self.pad = pad
        self.register_buffer("mean", torch.tensor(mean).view(1, 1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 1, 1, 1))

    def forward(self, x):
        if self.pad:
            x = F.pad(x, (self.pad,) * 4, mode="constant", value=0.0)
        return (x - self.mean) / self.std


def _activation(name):
    if name == "tanh":
        return nn.Tanh()
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"Unsupported activation: {name}")


class LeNet5(nn.Module):
    """LeNet-5 with dense C3 connections, faithful to ``legacy/src/models/lenet.py``.

    ``width`` scales every feature-map count, so ``width=2`` gives the wide
    teacher used for distillation in Phase 2.
    """

    def __init__(self, activation: str = "tanh", width: int = 1, num_classes: int = 10):
        super().__init__()
        c1, c3, c5, f6 = 6 * width, 16 * width, 120 * width, 84 * width

        self.normalize = InputNormalize()
        self.conv1 = nn.Conv2d(1, c1, kernel_size=5)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(c1, c3, kernel_size=5)
        self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(c3, c5, kernel_size=5)
        self.fc6 = nn.Linear(c5, f6)
        self.fc7 = nn.Linear(f6, num_classes)
        self.act = _activation(activation)

    def forward(self, x):
        x = self.normalize(x)
        x = self.pool2(self.act(self.conv1(x)))
        x = self.pool4(self.act(self.conv3(x)))
        x = self.act(self.conv5(x))
        x = torch.flatten(x, 1)
        x = self.act(self.fc6(x))
        return self.fc7(x)

    @property
    def prunable_modules(self):
        """(name, module) pairs Phase 2 magnitude pruning operates on."""
        return [
            ("conv1", self.conv1),
            ("conv3", self.conv3),
            ("conv5", self.conv5),
            ("fc6", self.fc6),
            ("fc7", self.fc7),
        ]


class MLP(nn.Module):
    """Fully-connected comparison model, ported from ``legacy/src/models/mlp.py``."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.normalize = InputNormalize()
        self.fc1 = nn.Linear(32 * 32, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.normalize(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    @property
    def prunable_modules(self):
        return [("fc1", self.fc1), ("fc2", self.fc2), ("fc3", self.fc3)]


def build_model(name: str = "lenet5", **kwargs) -> nn.Module:
    name = name.lower()
    if name in ("lenet5", "lenet"):
        return LeNet5(**kwargs)
    if name == "lenet5_wide":
        kwargs.setdefault("width", 2)
        return LeNet5(**kwargs)
    if name == "mlp":
        kwargs.pop("activation", None)
        kwargs.pop("width", None)
        return MLP(**kwargs)
    raise ValueError(f"Unknown model: {name}")
