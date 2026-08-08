"""One seed-controlled training run.

The run records validation and test accuracy at every epoch, but only the
validation curve is allowed to choose anything. Test accuracy is stored so that
Phase 0 can quantify the leakage in the legacy protocol -- reporting
best-epoch-on-test alongside best-epoch-on-val is the whole point of the
comparison. Downstream phases read ``test_acc_at_best_val`` and nothing else.
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from src.data import get_loaders
from src.models import build_model
from src.utils import RunRecord, config_hash, count_params, set_seed


@torch.no_grad()
def evaluate(model, loader, device, criterion=None):
    model.eval()
    criterion = criterion or nn.CrossEntropyLoss()
    loss_sum, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss_sum += criterion(out, y).item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        n += y.size(0)
    return loss_sum / n, 100.0 * correct / n


def train_one(
    seed: int = 0,
    epochs: int = 20,
    lr: float = 0.01,
    momentum: float = 0.9,
    batch_size: int = 64,
    model_name: str = "lenet5",
    activation: str = "tanh",
    optimizer_name: str = "sgd",
    augment: bool = False,
    data_dir: str = "./data",
    out_dir: str = None,
    tag: str = "baseline",
    smoke: bool = False,
    verbose: bool = True,
) -> RunRecord:
    if smoke:
        epochs = 1
    # Smoke records must never land in results/raw: they are 1-epoch/500-image
    # runs and would be averaged into the reported table as if they were real.
    if out_dir is None:
        out_dir = "results/smoke" if smoke else "results/raw"

    config = {
        "tag": tag,
        "model": model_name,
        "activation": activation,
        "optimizer": optimizer_name,
        "lr": lr,
        "momentum": momentum,
        "batch_size": batch_size,
        "epochs": epochs,
        "augment": augment,
        "smoke": smoke,
    }

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = get_loaders(batch_size=batch_size, data_dir=data_dir, augment=augment, seed=seed, smoke=smoke)

    model = build_model(model_name, activation=activation).to(device)
    if optimizer_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    elif optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    criterion = nn.CrossEntropyLoss()

    run_id = f"{tag}_{model_name}_{'aug' if augment else 'std'}_seed{seed}_{config_hash(config)}"
    record = RunRecord(run_id=run_id, config=config, seed=seed)

    best_val_acc, best_val_epoch = -1.0, -1
    best_state = None
    t0 = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for x, y in loaders.train:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item() * y.size(0)

        train_loss = running / len(loaders.train.dataset)
        val_loss, val_acc = evaluate(model, loaders.val, device, criterion)
        test_loss, test_acc = evaluate(model, loaders.test, device, criterion)

        record.epochs.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
            }
        )
        if verbose:
            print(
                f"  seed {seed} epoch {epoch:2d}/{epochs}  train_loss {train_loss:.4f}  "
                f"val_acc {val_acc:.2f}%  test_acc {test_acc:.2f}%",
                flush=True,
            )

        # Selection uses validation only.
        if val_acc > best_val_acc:
            best_val_acc, best_val_epoch = val_acc, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    record.wall_clock_sec = time.perf_counter() - t0

    # Correct protocol: the epoch chosen on validation, evaluated once on test.
    test_at_best_val = record.epochs[best_val_epoch - 1]["test_acc"]
    # Leaky protocol, reported only to quantify the bias in the legacy numbers.
    leaky_epoch = max(record.epochs, key=lambda e: e["test_acc"])

    record.metrics = {
        "best_val_acc": best_val_acc,
        "best_val_epoch": best_val_epoch,
        "test_acc_at_best_val": test_at_best_val,
        "final_epoch_test_acc": record.epochs[-1]["test_acc"],
        "leaky_best_test_acc": leaky_epoch["test_acc"],
        "leaky_best_epoch": leaky_epoch["epoch"],
        "leakage_gap": leaky_epoch["test_acc"] - test_at_best_val,
        **count_params(model),
    }

    path = record.save(out_dir)
    if verbose:
        m = record.metrics
        print(
            f"  seed {seed} done in {record.wall_clock_sec/60:.1f} min | "
            f"val-selected test {m['test_acc_at_best_val']:.2f}% (ep {best_val_epoch}) | "
            f"leaky test {m['leaky_best_test_acc']:.2f}% (ep {m['leaky_best_epoch']}) | "
            f"gap {m['leakage_gap']:+.2f} pp -> {path}",
            flush=True,
        )

    if best_state is not None and not smoke:
        ckpt_dir = Path("results/checkpoints")
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, ckpt_dir / f"{run_id}.pth")

    return record


def main():
    p = argparse.ArgumentParser(description="Train one seed-controlled run.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--model", dest="model_name", default="lenet5", choices=["lenet5", "lenet5_wide", "mlp"])
    p.add_argument("--activation", default="tanh", choices=["tanh", "relu"])
    p.add_argument("--optimizer", dest="optimizer_name", default="sgd", choices=["sgd", "adam"])
    p.add_argument("--augment", action="store_true")
    p.add_argument("--tag", default="baseline")
    p.add_argument("--out-dir", default=None, help="default: results/raw, or results/smoke under --smoke")
    p.add_argument("--smoke", action="store_true", help="1 epoch, 500 images -- correctness check only")
    args = p.parse_args()
    train_one(**vars(args))


if __name__ == "__main__":
    main()
