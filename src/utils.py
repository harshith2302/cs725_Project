"""Seeding, provenance and run logging.

Methodology rule 5 of PROJECT_PLAN.md: no metric appears in a figure unless it
came from a JSON file on disk. ``RunRecord`` is that file. Every run writes one,
carrying enough provenance (git SHA, config hash, library versions, machine
spec) to tell later whether two numbers are actually comparable.
"""

import hashlib
import json
import os
import platform
import random
import subprocess
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def git_sha(short: bool = False) -> str:
    cmd = ["git", "rev-parse", "--short" if short else "HEAD"]
    try:
        sha = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
    try:
        dirty = subprocess.check_output(["git", "status", "--porcelain"]).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        dirty = ""
    return f"{sha}-dirty" if dirty else sha


def config_hash(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()[:12]


def machine_spec() -> dict:
    """Recorded because Phase 1 latency numbers are meaningless without it."""
    return {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def count_params(model: torch.nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    nonzero = sum(int((p != 0).sum()) for p in model.parameters())
    weights = [m.weight for _, m in getattr(model, "prunable_modules", [])]
    prunable = sum(w.numel() for w in weights)
    prunable_nonzero = sum(int((w != 0).sum()) for w in weights)
    return {
        "total_params": total,
        "nonzero_params": nonzero,
        "prunable_params": prunable,
        "prunable_nonzero_params": prunable_nonzero,
        "sparsity": 1.0 - (prunable_nonzero / prunable) if prunable else 0.0,
    }


@dataclass
class RunRecord:
    """One JSON file per run. Written atomically at the end of the run."""

    run_id: str
    config: dict
    seed: int
    git_sha: str = field(default_factory=git_sha)
    machine: dict = field(default_factory=machine_spec)
    metrics: dict = field(default_factory=dict)
    epochs: list = field(default_factory=list)
    wall_clock_sec: float = 0.0
    started_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))

    def save(self, out_dir="results/raw") -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{self.run_id}.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2, default=str), encoding="utf-8")
        tmp.replace(path)
        return path


def load_records(out_dir="results/raw", prefix: str = "") -> list:
    d = Path(out_dir)
    if not d.exists():
        return []
    return [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(d.glob("*.json"))
        if p.name.startswith(prefix)
    ]


def mean_std(values) -> tuple:
    a = np.asarray(values, dtype=float)
    # ddof=1: sample std across seeds, which is what "mean +/- std over 5 seeds" means.
    return float(a.mean()), float(a.std(ddof=1)) if a.size > 1 else 0.0
