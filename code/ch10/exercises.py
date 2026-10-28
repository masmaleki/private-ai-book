"""exercises.py - runnable answers for exercises 10-6 and 10-7 (example data)."""
import numpy as np
import pandas as pd
from metrics import wape, bias
from report import bootstrap_mae_diff  # note: importing runs report.py once


def segment_report(df, y_col, base_col, model_col, seg_col):
    rows = []
    for seg, g in df.groupby(seg_col):
        wb = wape(g[y_col], g[base_col])
        wm = wape(g[y_col], g[model_col])
        rows.append({"segment": seg, "n": len(g), "wape_base": round(wb, 3),
                     "wape_model": round(wm, 3),
                     "bias_model": round(bias(g[y_col], g[model_col]), 3),
                     "worse": wm > wb})
    return pd.DataFrame(rows)


def cluster_bootstrap_mae_diff(y, a, b, groups, n_boot=2000, seed=7):
    """95% CI for MAE(a) - MAE(b), resampling whole groups (e.g. rfq_id)."""
    r = np.random.default_rng(seed)
    y, a, b, groups = map(np.asarray, (y, a, b, groups))
    uniq = np.unique(groups)
    idx_by_group = [np.where(groups == g)[0] for g in uniq]
    diffs = []
    for _ in range(n_boot):
        pick = r.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([idx_by_group[k] for k in pick])
        diffs.append(np.abs(y[idx] - a[idx]).mean() - np.abs(y[idx] - b[idx]).mean())
    return np.percentile(diffs, [2.5, 97.5])


if __name__ == "__main__":
    print("\n=== exercise 10-6 ===")
    df = pd.DataFrame({"mfr": ["A", "A", "B", "B"], "y": [1.0, 2.0, 1.0, 1.0],
                       "base": [1.1, 1.8, 1.0, 1.1], "model": [1.0, 2.0, 1.3, 1.2]})
    print(segment_report(df, "y", "base", "model", "mfr"))

    print("\n=== exercise 10-7 ===")
    rng = np.random.default_rng(1)
    groups = np.repeat(np.arange(100), 10)
    rfq_effect = rng.normal(0, 0.2, 100)[groups]
    y = np.ones(1000)
    a = y + rfq_effect + rng.normal(0, 0.01, 1000)
    b = y + 1.1 * rfq_effect + rng.normal(0, 0.01, 1000)
    row = bootstrap_mae_diff(y, a, b)
    clu = cluster_bootstrap_mae_diff(y, a, b, groups)
    print("row-level CI:", np.round(row, 4), "width", round(row[1] - row[0], 4))
    print("cluster CI:  ", np.round(clu, 4), "width", round(clu[1] - clu[0], 4))
