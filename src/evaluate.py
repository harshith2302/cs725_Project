"""Measurement suite: clean accuracy, MNIST-C, latency, model size.

Built before any compression method, because every downstream claim depends on
these being correct.
"""

import os
import statistics
import tempfile
import time

import torch
import torch.nn as nn

from src.utils import count_params, machine_spec


@torch.no_grad()
def clean_accuracy(model, loader, device) -> float:
    model.eval()
    correct = n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        n += y.size(0)
    return 100.0 * correct / n


@torch.no_grad()
def mnist_c_accuracy(model, corruption_loaders: dict, device) -> dict:
    """Per-corruption accuracy plus the mean across corruptions.

    The mean alone hides which corruptions break which models, so both are
    always reported (PROJECT_PLAN.md Phase 1.1).
    """
    per = {name: clean_accuracy(model, ld, device) for name, ld in sorted(corruption_loaders.items())}
    accs = list(per.values())
    return {
        "per_corruption": per,
        "mean_acc": sum(accs) / len(accs),
        "worst_corruption": min(per, key=per.get),
        "worst_acc": min(accs),
        "n_corruptions": len(per),
    }


@torch.no_grad()
def latency(model, device=None, warmup: int = 100, iters: int = 1000, single_thread: bool = True) -> dict:
    """Batch-1 CPU latency (median) and batch-128 throughput.

    Median rather than mean: a single scheduler hiccup skews the mean, and the
    typical-case number is what a deployment cares about. Machine spec travels
    with the measurement because latency without hardware context is noise.
    """
    device = device or torch.device("cpu")
    prev_threads = torch.get_num_threads()
    if single_thread and device.type == "cpu":
        torch.set_num_threads(1)
    model = model.to(device).eval()

    try:
        x1 = torch.rand(1, 1, 28, 28, device=device)
        for _ in range(warmup):
            model(x1)
        samples = []
        for _ in range(iters):
            t0 = time.perf_counter()
            model(x1)
            samples.append((time.perf_counter() - t0) * 1000.0)

        x128 = torch.rand(128, 1, 28, 28, device=device)
        for _ in range(10):
            model(x128)
        t0 = time.perf_counter()
        for _ in range(50):
            model(x128)
        batch_s = (time.perf_counter() - t0) / 50
    finally:
        torch.set_num_threads(prev_threads)

    samples.sort()
    return {
        "latency_ms_median": statistics.median(samples),
        "latency_ms_mean": statistics.fmean(samples),
        "latency_ms_p90": samples[int(0.90 * len(samples))],
        "latency_ms_p99": samples[int(0.99 * len(samples))],
        "throughput_img_per_s_bs128": 128.0 / batch_s,
        "batch_size_1_iters": iters,
        "warmup_iters": warmup,
        "single_thread": single_thread and device.type == "cpu",
        "machine": machine_spec(),
    }


def _file_size_kb(save_fn) -> float:
    fd, path = tempfile.mkstemp(suffix=".pth")
    os.close(fd)
    try:
        save_fn(path)
        return os.path.getsize(path) / 1024.0
    finally:
        os.unlink(path)


def model_size(model) -> dict:
    """Parameter counts and honest on-disk sizes.

    Unstructured sparsity does **not** shrink a dense checkpoint: a pruned model
    saved with torch.save is byte-for-byte the same size as the dense one, because
    the zeros are still stored. Both the real dense file size and the theoretical
    CSR size are reported; quoting only the latter as "the model size" would be a
    misrepresentation.
    """
    counts = count_params(model)
    fp32_kb = _file_size_kb(lambda p: torch.save(model.state_dict(), p))

    # CSR for the prunable weight matrices: values (4B) + column indices (4B)
    # per non-zero, plus a row-pointer array. Everything else stays dense.
    csr_bytes = 0
    for _, m in getattr(model, "prunable_modules", []):
        w = m.weight.detach()
        nnz = int((w != 0).sum())
        rows = w.shape[0]
        csr_bytes += nnz * (4 + 4) + (rows + 1) * 4
    dense_other = sum(
        p.numel() * 4
        for name, p in model.named_parameters()
        if not any(name == f"{n}.weight" for n, _ in getattr(model, "prunable_modules", []))
    )

    return {
        **counts,
        "fp32_checkpoint_kb": fp32_kb,
        "theoretical_csr_kb": (csr_bytes + dense_other) / 1024.0,
        "note": (
            "fp32_checkpoint_kb is the real file on disk; unstructured sparsity does not "
            "shrink it. theoretical_csr_kb is what a sparse format would cost and is not "
            "achieved by torch.save."
        ),
    }


def evaluate_all(model, loaders, device, corruption_loaders=None, attack_kwargs=None) -> dict:
    """The six-column row for one configuration."""
    from src.attacks import adversarial_accuracy

    out = {
        "clean_acc": clean_accuracy(model, loaders.test, device),
        "size": model_size(model),
        "latency": latency(model, torch.device("cpu")),
    }
    if corruption_loaders:
        out["mnist_c"] = mnist_c_accuracy(model, corruption_loaders, device)
    out["adversarial"] = adversarial_accuracy(model, loaders.test, device, **(attack_kwargs or {}))
    return out
