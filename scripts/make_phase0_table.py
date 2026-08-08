"""Build the Phase 0 table from results/raw/*.json.

Kept separate from the runner so the table is always regenerated from the JSON
records on disk, never from whatever happened to be in memory during one
invocation (PROJECT_PLAN.md methodology rule 5). Running only one condition
therefore cannot silently drop the other from the table.

    python scripts/make_phase0_table.py
"""

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import load_records, mean_std

CONDITIONS = ["standard", "augmented"]

# The Windows console defaults to cp1252, which cannot encode this table's
# typography. Done at import so callers (e.g. run_phase0.py) get it too; the
# markdown file itself is always written as UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROTOCOLS = [
    ("test_acc_at_best_val", "Correct - best epoch on validation"),
    ("leaky_best_test_acc", "Leaky - best epoch on test"),
    ("final_epoch_test_acc", "Final epoch (no selection)"),
]


def welch(a, b):
    """Welch's t-test, two-sided. Implemented here to avoid a scipy dependency
    for a four-line formula; Phase 4 uses scipy.stats and cross-checks this."""
    na, nb = len(a), len(b)
    ma, sa = mean_std(a)
    mb, sb = mean_std(b)
    va, vb = sa**2 / na, sb**2 / nb
    se = math.sqrt(va + vb)
    if se == 0:
        return 0.0, float("nan"), float("nan")
    t = (mb - ma) / se
    df = (va + vb) ** 2 / (va**2 / (na - 1) + vb**2 / (nb - 1))
    # Two-sided p from the t CDF via the regularised incomplete beta function.
    x = df / (df + t * t)
    p = _betainc(df / 2.0, 0.5, x)
    return t, df, p


def _betainc(a, b, x):
    """Regularised incomplete beta I_x(a, b), continued-fraction form."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1 - x) * b - lbeta) / a
    if x >= (a + 1) / (a + b + 2):
        return 1.0 - _betainc(b, a, 1 - x)
    f, c, d = 1.0, 1.0, 0.0
    for i in range(200):
        m = i // 2
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m))
        else:
            num = -((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 / (1.0 + num * d) if abs(1.0 + num * d) > 1e-30 else 1e30
        c = 1.0 + num / c if abs(num / c) < 1e30 else 1e30
        f *= c * d
        if abs(1.0 - c * d) < 1e-10:
            break
    return front * (f - 1.0)


def fmt(values):
    m, s = mean_std(values)
    return f"{m:.2f} ± {s:.2f}"


def truncate_budget(record, budget):
    """Re-derive a shorter training budget from a stored epoch curve.

    Valid only because these runs use a constant learning rate with no
    scheduler: with the same seed, the first `budget` epochs of a 20-epoch run
    are bit-identical to a `budget`-epoch run. Lets Phase 0 compare against the
    legacy 10-epoch protocol without retraining. Do not reuse this for any
    config with an LR schedule.
    """
    epochs = record["epochs"][:budget]
    best_val = max(epochs, key=lambda e: e["val_acc"])
    best_test = max(epochs, key=lambda e: e["test_acc"])
    return {
        "test_acc_at_best_val": best_val["test_acc"],
        "leaky_best_test_acc": best_test["test_acc"],
        "final_epoch_test_acc": epochs[-1]["test_acc"],
        "leakage_gap": best_test["test_acc"] - best_val["test_acc"],
    }


def build(records_dir="results/raw"):
    recs = {}
    for c in CONDITIONS:
        found = load_records(records_dir, f"phase0-{c}")
        # Second guard against smoke contamination: train_one already routes
        # smoke runs to results/smoke, but a stray record must not be averaged
        # into a reported number just because it sits in the right directory.
        real = [r for r in found if not r["config"].get("smoke")]
        if len(real) < len(found):
            print(
                f"warning: ignoring {len(found) - len(real)} smoke record(s) in {records_dir} "
                f"for condition '{c}'",
                file=sys.stderr,
            )
        if real:
            recs[c] = sorted(real, key=lambda r: r["seed"])
    if not recs:
        sys.exit(f"no phase0 records found in {records_dir}")

    budgets = {r["config"]["epochs"] for rs in recs.values() for r in rs}
    if len(budgets) > 1:
        sys.exit(f"records mix training budgets {sorted(budgets)}; refusing to average across them")

    any_rec = next(iter(recs.values()))[0]
    cfg = any_rec["config"]
    L = [
        "# Phase 0 — leakage-corrected baseline",
        "",
        f"LeNet-5 (dense C3, {any_rec['metrics']['total_params']:,} params), "
        f"{cfg['optimizer']} lr={cfg['lr']} momentum={cfg['momentum']}, "
        f"batch {cfg['batch_size']}, {cfg['epochs']} epochs.",
        "Split 55,000 train / 5,000 val / 10,000 test, `SPLIT_SEED=1234` fixed across seeds.",
        f"Machine: {any_rec['machine']['platform']}, torch {any_rec['machine']['torch']}, "
        f"CUDA {any_rec['machine']['cuda_available']}.",
        f"Generated from `{records_dir}/*.json` at commit `{any_rec['git_sha']}`.",
        "",
        "## Test accuracy (%), mean ± std over seeds",
        "",
        "| Protocol | " + " | ".join(f"{c} (n={len(r)})" for c, r in recs.items()) + " |",
        "| --- |" + " --- |" * len(recs),
    ]
    for key, label in PROTOCOLS:
        row = [fmt([r["metrics"][key] for r in rs]) for rs in recs.values()]
        L.append(f"| {label} | " + " | ".join(row) + " |")

    L += [
        "",
        "## Size of the test-set leakage",
        "",
        "Per-seed difference between the two selection protocols on the *same* run.",
        "",
        "| Condition | Leaky − correct (pp) | per-seed |",
        "| --- | --- | --- |",
    ]
    for c, rs in recs.items():
        gaps = [r["metrics"]["leakage_gap"] for r in rs]
        L.append(f"| {c} | {fmt(gaps)} | {', '.join(f'{g:+.2f}' for g in gaps)} |")

    L += ["", "## Per-seed detail", "", "| Condition | Seed | Best val epoch | Val acc | Test @ best val | Leaky test |", "| --- | --- | --- | --- | --- | --- |"]
    for c, rs in recs.items():
        for r in rs:
            m = r["metrics"]
            L.append(
                f"| {c} | {r['seed']} | {m['best_val_epoch']} | {m['best_val_acc']:.2f} | "
                f"{m['test_acc_at_best_val']:.2f} | {m['leaky_best_test_acc']:.2f} |"
            )

    if len(recs) == 2:
        L += [
            "",
            "## Is the augmentation gain real?",
            "",
            "Welch's t-test across the 5 seeds, two-sided. Both training budgets and both",
            "selection protocols are reported: which cell you look at changes the answer,",
            "which is itself the finding.",
            "",
            "| Budget | Protocol | standard | augmented | diff (pp) | t | df | p | verdict |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for budget in (10, 20):
            for key, label in PROTOCOLS[:2]:
                if budget == 20:
                    a = [r["metrics"][key] for r in recs["standard"]]
                    b = [r["metrics"][key] for r in recs["augmented"]]
                else:
                    a = [truncate_budget(r, budget)[key] for r in recs["standard"]]
                    b = [truncate_budget(r, budget)[key] for r in recs["augmented"]]
                ma, sa = mean_std(a)
                mb, sb = mean_std(b)
                t, df, p = welch(a, b)
                verdict = "**not sig.**" if p > 0.05 else "sig."
                short = "best val" if "validation" in label else "best test (leaky)"
                L.append(
                    f"| {budget} ep | {short} | {ma:.2f} ± {sa:.2f} | {mb:.2f} ± {sb:.2f} | "
                    f"{mb - ma:+.2f} | {t:.2f} | {df:.1f} | {p:.3f} | {verdict} |"
                )
        L.append("")
        L += [
            "For reference, the original single-seed write-up reported 99.01 (standard) vs "
            "99.12 (augmented), a +0.11 pp gap, and treated it as a real effect.",
            "",
        ]

    L += [
        "## Budget-matched against the legacy 10-epoch protocol",
        "",
        "The legacy runs trained for 10 epochs; these for 20. Because the learning rate is",
        "constant with no scheduler, the first 10 epochs of each run are identical to a",
        "10-epoch run at the same seed, so the shorter budget is re-derived exactly from the",
        "stored curves rather than retrained (`truncate_budget`).",
        "",
        "| Condition | Budget | Correct (best val) | Leaky (best test) | Legacy reported |",
        "| --- | --- | --- | --- | --- |",
    ]
    legacy_reported = {"standard": "99.01", "augmented": "99.12"}
    for c, rs in recs.items():
        for budget in (10, 20):
            t = [truncate_budget(r, budget) for r in rs]
            ref = legacy_reported.get(c, "—") if budget == 10 else "—"
            L.append(
                f"| {c} | {budget} epochs | {fmt([x['test_acc_at_best_val'] for x in t])} | "
                f"{fmt([x['leaky_best_test_acc'] for x in t])} | {ref} |"
            )
    L += [
        "",
        "The legacy numbers were produced under the *leaky* 10-epoch protocol, so that is the",
        "column they should be read against.",
        "",
    ]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--records-dir", default="results/raw")
    ap.add_argument("--out", default="results/tables/phase0_baseline.md")
    args = ap.parse_args()
    text = build(args.records_dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"wrote {out}")
