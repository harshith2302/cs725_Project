import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import os
import numpy as np

def get_data_loaders(batch_size=64, data_dir='./data', augment=False, train_pct=1.0):
    """
    Creates and returns the MNIST data loaders.
    """
    # Base transform (Resize/Pad + Normalize)
    base_transform = [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]

    if augment:
        # Add distortions if augmentation is enabled
        train_transform = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        ] + base_transform)
    else:
        train_transform = transforms.Compose(base_transform)

    test_transform = transforms.Compose(base_transform)

    # Ensure data directory exists
    os.makedirs(data_dir, exist_ok=True)

    train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=train_transform)
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=test_transform)

    # Handle Training Set Size
    if train_pct < 1.0:
        num_train = len(train_dataset)
        indices = list(range(num_train))
        split = int(np.floor(train_pct * num_train))
        np.random.shuffle(indices)
        train_idx = indices[:split]
        train_dataset = Subset(train_dataset, train_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    return train_loader, test_loader
