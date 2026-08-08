"""Phase 0 baseline: 5 seeds, standard and augmented, both selection protocols.

Writes one JSON per run to results/raw/ and a markdown summary to
results/tables/phase0_baseline.md.

    python scripts/run_phase0.py --smoke      # correctness check, seconds
    python scripts/run_phase0.py              # the real thing
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_phase0_table import build as build_table
from src.train import train_one

SEEDS = [0, 1, 2, 3, 4]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--conditions", nargs="+", default=["standard", "augmented"])
    args = p.parse_args()

    for cond in args.conditions:
        print(f"\n=== condition: {cond} ===", flush=True)
        for seed in args.seeds:
            train_one(
                seed=seed,
                epochs=args.epochs,
                augment=(cond == "augmented"),
                tag=f"phase0-{cond}",
                smoke=args.smoke,
            )

    # Rebuilt from every phase0 record on disk, so running one condition at a
    # time still produces the complete table.
    out = Path("results/tables/phase0_baseline.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    text = build_table()
    out.write_text(text, encoding="utf-8")
    print("\n" + text)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
