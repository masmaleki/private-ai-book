"""Chapter 9: rolling-origin backtest of simple methods vs a global LightGBM.

Forecast origin T = end of a month; horizons h = 1..3 months ahead.
Every method sees only data up to and including month T (synthetic data).
"""
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd

from demand import make

warnings.filterwarnings("ignore")
H = (1, 2, 3)
ORIGINS = range(35, 45)          # month indexes 35..44 -> targets up to index 47


def wide():
    d = make()
    Y = d.pivot(index="part", columns="month", values="qty").to_numpy(float)
    kind = d.groupby("part").kind.first().to_numpy()
    return Y, kind


def croston_sba(y, alpha=0.1):
    nz = np.flatnonzero(y > 0)
    if len(nz) == 0:
        return 0.0
    z, p, last = y[nz[0]], nz[0] + 1.0, nz[0]
    for t in nz[1:]:
        z += alpha * (y[t] - z)
        p += alpha * ((t - last) - p)
        last = t
    return (1 - alpha / 2) * z / p


def seasonal_index(hist):
    """Global monthly index from the pooled total of all parts (last 24 months)."""
    tot = hist.sum(0)[-24:]
    T = hist.shape[1] - 1
    months = np.arange(T - 23, T + 1) % 12
    ratio = tot / tot.mean()
    return np.array([ratio[months == m].mean() for m in range(12)])


def simple_forecasts(Y, T, h):
    hist = Y[:, : T + 1]
    idx = seasonal_index(hist)
    return {
        "zero": np.zeros(len(Y)),
        "naive": hist[:, -1],
        "seasonal naive": Y[:, T + h - 12],
        "moving avg 3": hist[:, -3:].mean(1),
        "moving avg 12": hist[:, -12:].mean(1),
        "ma12 x season": hist[:, -12:].mean(1) * idx[(T + h) % 12],
        "croston sba": np.array([croston_sba(r) for r in hist]),
    }


def features(Y, t, h):
    """Features known at origin t for target t + h, scaled per series."""
    hist = Y[:, : t + 1]
    scale = hist[:, -12:].mean(1) + 1.0
    lags = hist[:, -12:][:, ::-1] / scale[:, None]          # lag0 .. lag11
    f = {f"lag{k}": lags[:, k] for k in range(12)}
    f["ma3"] = hist[:, -3:].mean(1) / scale
    f["zero_share12"] = (hist[:, -12:] == 0).mean(1)
    f["same_month_ly"] = Y[:, t + h - 12] / scale
    f["target_month"] = np.full(len(Y), (t + h) % 12)
    f["h"] = np.full(len(Y), h)
    return pd.DataFrame(f), scale


def train_rows(Y, T):
    X, y = [], []
    for t in range(12, T):
        for h in H:
            if t + h > T:      # target must be known at origin T: no leakage
                continue
            f, s = features(Y, t, h)
            X.append(f)
            y.append(Y[:, t + h] / s)
    return pd.concat(X, ignore_index=True), np.concatenate(y)


def gbm(alpha=None, objective="l1"):
    kw = dict(n_estimators=300, learning_rate=0.05, num_leaves=15, objective=objective,
              min_child_samples=20, verbose=-1, random_state=0, n_jobs=2)
    if alpha is not None:
        kw.update(objective="quantile", alpha=alpha)
    return lgb.LGBMRegressor(**kw)


def run():
    Y, kind = wide()
    recs = []
    for T in ORIGINS:
        X, y = train_rows(Y, T)
        models = {"lightgbm global": gbm().fit(X, y),
                  "lightgbm (l2)": gbm(objective="l2").fit(X, y),
                  "p10": gbm(0.1).fit(X, y), "p90": gbm(0.9).fit(X, y)}
        for h in H:
            if T + h >= Y.shape[1]:
                continue
            preds = simple_forecasts(Y, T, h)
            f, s = features(Y, T, h)
            for name in ("lightgbm global", "lightgbm (l2)"):
                preds[name] = np.clip(models[name].predict(f), 0, None) * s
            lo = np.clip(models["p10"].predict(f), 0, None) * s
            hi = np.clip(models["p90"].predict(f), 0, None) * s
            for name, p in preds.items():
                recs.append(pd.DataFrame({"origin": T, "h": h, "method": name,
                                          "kind": kind, "part_i": np.arange(len(Y)),
                                          "y": Y[:, T + h], "p": p,
                                          "lo": lo, "hi": hi}))
    return pd.concat(recs, ignore_index=True)


def wape(g):
    return 100 * np.abs(g.y - g.p).sum() / g.y.sum()


if __name__ == "__main__":
    r = run()
    print(f"forecasts evaluated: {len(r[r.method == 'naive'])} per method")
    tab = r.groupby(["method", "kind"]).apply(wape).unstack().round(1)
    tab["all"] = r.groupby("method").apply(wape).round(1)
    print("WAPE % by demand kind (lower is better):")
    print(tab[["smooth", "intermittent", "lumpy", "all"]].to_string())
    print("\nWAPE % by horizon, smooth parts:")
    sm = r[r.kind == "smooth"]
    print(sm.groupby(["method", "h"]).apply(wape).unstack().round(1).to_string())
    tot = r.groupby(["method", "kind", "origin", "part_i"])[["y", "p"]].sum().reset_index()
    print("\nWAPE % of the 3-month total (what purchasing needs):")
    t3 = tot.groupby(["method", "kind"]).apply(wape).unstack().round(1)
    print(t3[["smooth", "intermittent", "lumpy"]].to_string())
    g = r[r.method == "lightgbm global"]
    inside = (g.y >= g.lo) & (g.y <= g.hi)
    print("\ncoverage of [P10, P90] by kind:")
    print(inside.groupby(g.kind).mean().round(3).to_string())
    print(f"overall coverage: {inside.mean():.3f}")
    r.to_csv("backtest_results.csv", index=False)
