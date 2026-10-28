"""report.py - evaluation report + release gate on synthetic data (example data only).
Compares a candidate price model against a baseline, per segment, with a bootstrap
confidence interval, then checks release-gate rules and writes report.md.
"""
import numpy as np
from metrics import mae, wape, bias, coverage, pinball, roc_auc, pr_auc, brier, ece

rng = np.random.default_rng(42)
N = 2000

# ---- synthetic test set (example data, not real company data) ----
manufacturer = rng.choice(["ST", "TI", "Murata", "Microchip"], size=N, p=[.35, .35, .2, .1])
qty = rng.choice([100, 500, 1000, 5000, 20000], size=N)
tenant = rng.choice(["tenant_A", "tenant_B", "tenant_C"], size=N)
true_price = np.exp(rng.normal(0.0, 0.6, N)) * (qty / 1000.0) ** -0.08

# Baseline: last observed price, noisy. Candidate: better on average, worse for Microchip
baseline = true_price * np.exp(rng.normal(0, 0.18, N))
cand_noise = np.where(manufacturer == "Microchip", 0.25, 0.10)
candidate = true_price * np.exp(rng.normal(0.01, cand_noise))
p10 = candidate * np.exp(-1.2816 * cand_noise)   # model's own interval
p90 = candidate * np.exp(+1.2816 * cand_noise)

# Win-probability model on the same quotes (synthetic labels)
logit = rng.normal(0, 1.2, N)
won = (rng.random(N) < 1 / (1 + np.exp(-logit))).astype(int)
p_win = 1 / (1 + np.exp(-(0.7 * logit + 0.4)))      # deliberately over-confident

def bootstrap_mae_diff(y, a, b, n_boot=2000, seed=7):
    """95% CI for MAE(a) - MAE(b) by resampling rows."""
    r = np.random.default_rng(seed)
    idx = r.integers(0, len(y), size=(n_boot, len(y)))
    d = np.abs(y[idx] - a[idx]).mean(1) - np.abs(y[idx] - b[idx]).mean(1)
    return np.percentile(d, [2.5, 97.5])

lines = ["# Evaluation report - price model v2 (synthetic example)", ""]
lines.append(f"Test rows: {N}")
lines.append("")
lines.append("| segment | n | WAPE base | WAPE cand | bias cand |")
lines.append("|---|---:|---:|---:|---:|")

segments = [("ALL", np.ones(N, bool))]
segments += [(f"mfr={m}", manufacturer == m) for m in ["ST", "TI", "Murata", "Microchip"]]
segments += [(f"qty={q}", qty == q) for q in [100, 1000, 20000]]
segments += [(f"{t}", tenant == t) for t in ["tenant_A", "tenant_B", "tenant_C"]]
worse = []
for name, m in segments:
    wb, wc = wape(true_price[m], baseline[m]), wape(true_price[m], candidate[m])
    lines.append(f"| {name} | {m.sum()} | {wb:.3f} | {wc:.3f} | {bias(true_price[m], candidate[m]):+.3f} |")
    if wc > wb:
        worse.append(name)

lo, hi = bootstrap_mae_diff(true_price, candidate, baseline)
cov = coverage(true_price, p10, p90)
lines += ["", f"MAE(cand) - MAE(base), 95% bootstrap CI: [{lo:+.4f}, {hi:+.4f}]",
          f"P10-P90 coverage: {cov:.3f} (target 0.80)",
          f"Pinball P90: {pinball(true_price, p90, 0.9):.4f}",
          "", "Win-probability model:",
          f"ROC-AUC {roc_auc(won, p_win):.3f} | PR-AUC {pr_auc(won, p_win):.3f} | "
          f"Brier {brier(won, p_win):.3f} | ECE {ece(won, p_win):.3f}"]

# ---- release gate: all rules must pass ----
gate = {
    "beats baseline overall (CI upper bound < 0)": hi < 0,
    "no segment worse than baseline": not worse,
    "interval coverage within 0.75-0.85": 0.75 <= cov <= 0.85,
    "win model ECE <= 0.05": ece(won, p_win) <= 0.05,
}
lines += ["", "## Release gate"]
for rule, ok in gate.items():
    lines.append(f"- [{'PASS' if ok else 'FAIL'}] {rule}")
if worse:
    lines.append(f"- segments worse than baseline: {', '.join(worse)}")
decision = "RELEASE" if all(gate.values()) else "BLOCK (fix or limit scope)"
lines.append(f"\nDecision: {decision}")

open("report.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
