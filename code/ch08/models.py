"""Chapter 8: linear vs random forest vs LightGBM, quantiles and SHAP.

Uses the synthetic demo_* tables and point-in-time features from chapter 7.
"""
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ch07"))
from experiments import BASE, CUTOFF, prepare  # noqa: E402
from features import build, load, offers_in_eur, qty_slope  # noqa: E402

warnings.filterwarnings("ignore")
FEATS = BASE + ["n30"]


def mape(y_log, p_log):
    y, p = np.exp(y_log), np.exp(p_log)
    return float(np.mean(np.abs(p - y) / y) * 100)


def data():
    t = load()
    o = offers_in_eur(t["demo_offers"], t["fx_rates"])
    slope = qty_slope(o[o.observed_at < CUTOFF])
    q = prepare(build(t, slope=slope))
    # "market" = 30-day median adjusted to the quote's quantity (log space)
    q["market"] = q["log_med30"] + slope * (q["log10_qty"] - 3)
    q["rel"] = q["y"] - q["market"]          # relative target: log(price / market)
    return q[q.issued_at < CUTOFF], q[q.issued_at >= CUTOFF]


def compare(tr, te):
    print("== 1) temporal split, MAPE on the test period ==")
    markup = float(np.median(tr["rel"]))
    base = mape(te.y, te.market + markup)
    print(f"{'baseline (market + median markup)':<40} {base:5.2f}%")
    models = {
        "linear": LinearRegression(),
        "random forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                                               random_state=0, n_jobs=1),
        "lightgbm": lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03,
                                      num_leaves=15, min_child_samples=20,
                                      subsample=0.8, subsample_freq=1,
                                      colsample_bytree=0.8, verbose=-1, n_jobs=2,
                                      random_state=0),
    }
    for name, m in models.items():
        m.fit(tr[FEATS], tr["y"])
        err = mape(te.y, m.predict(te[FEATS]))
        print(f"{name + ', target log(price)':<40} {err:5.2f}%")
    for name, m in models.items():
        m.fit(tr[FEATS], tr["rel"])
        pred = te.market + m.predict(te[FEATS])
        print(f"{name + ', target log(price/market)':<40} {mape(te.y, pred):5.2f}%")


def quantiles(tr, te):
    print("\n== 2) quantile LightGBM: P10 / P50 / P90 of log(price/market) ==")
    preds = {}
    for a in (0.1, 0.5, 0.9):
        m = lgb.LGBMRegressor(objective="quantile", alpha=a, n_estimators=300,
                              learning_rate=0.03, num_leaves=15, min_child_samples=20,
                              verbose=-1, random_state=0, n_jobs=2)
        m.fit(tr[FEATS], tr["rel"])
        preds[a] = te.market.to_numpy() + m.predict(te[FEATS])
    p10, p50, p90 = (np.exp(preds[a]) for a in (0.1, 0.5, 0.9))
    y = np.exp(te.y.to_numpy())
    inside = (y >= p10) & (y <= p90)
    print(f"coverage of [P10, P90]: {inside.mean():.1%}  (target 80%)")
    print(f"below P10: {(y < p10).mean():.1%}  above P90: {(y > p90).mean():.1%}")
    print(f"median width (P90-P10)/P50: {np.median((p90 - p10) / p50):.1%}")
    print(f"crossed quantiles (P10 > P90): {(p10 > p90).sum()} rows")
    print(f"P50 MAPE: {np.mean(np.abs(p50 - y) / y) * 100:.2f}%")
    i = 0
    print(f"example quote {int(te.quote_line_id.iloc[i])}: P10={p10[i]:.3f}  "
          f"P50={p50[i]:.3f}  P90={p90[i]:.3f}  actual={y[i]:.3f} EUR")
    return p10, p50, p90


def calibrated(tr, te):
    """Split-conformal widening: fit on early train, measure misses on late train."""
    print("\n== 3) calibrate the band on the last 3 months of training ==")
    cal_start = CUTOFF - pd.DateOffset(months=3)
    fit, cal = tr[tr.issued_at < cal_start], tr[tr.issued_at >= cal_start]
    lo_hi = {}
    for a in (0.1, 0.9):
        m = lgb.LGBMRegressor(objective="quantile", alpha=a, n_estimators=300,
                              learning_rate=0.03, num_leaves=15, min_child_samples=20,
                              verbose=-1, random_state=0, n_jobs=2)
        m.fit(fit[FEATS], fit["rel"])
        lo_hi[a] = (m.predict(cal[FEATS]), m.predict(te[FEATS]))
    # conformity score: how far outside the band each calibration point falls
    score = np.maximum(lo_hi[0.1][0] - cal["rel"], cal["rel"] - lo_hi[0.9][0])
    n = len(score)
    k = np.quantile(score, min(1.0, np.ceil((n + 1) * 0.8) / n))
    lo = np.exp(te.market + lo_hi[0.1][1] - k)
    hi = np.exp(te.market + lo_hi[0.9][1] + k)
    y = np.exp(te.y)
    print(f"calibration rows={n}  widening k={k:+.4f} (log units)")
    print(f"coverage after calibration: {((y >= lo) & (y <= hi)).mean():.1%}")
    print(f"median width after: {np.median((hi - lo) / np.sqrt(hi * lo)):.1%}")


def explain(tr, te):
    print("\n== 4) SHAP: why this prediction? (P50-like model, relative target) ==")
    m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15,
                          min_child_samples=20, verbose=-1, random_state=0, n_jobs=2)
    m.fit(tr[FEATS], tr["rel"])
    ex = shap.TreeExplainer(m)
    sv = ex.shap_values(te[FEATS])
    imp = pd.Series(np.abs(sv).mean(axis=0), index=FEATS).sort_values(ascending=False)
    print("mean |SHAP| (log units):")
    print(imp.round(4).to_string())
    i = 0
    row = pd.Series(sv[i], index=FEATS).sort_values(key=np.abs, ascending=False)
    qid = int(te.quote_line_id.iloc[i])
    print(f"\nquote {qid}: base value = {ex.expected_value:.4f}")
    for f, v in row.head(4).items():
        print(f"  {f:<14} = {te[f].iloc[i]:8.3f}  contribution {v:+.4f}")
    total = ex.expected_value + sv[i].sum()
    out = m.predict(te[FEATS].iloc[[i]])[0]
    print(f"  sum = {total:.4f}  model output = {out:.4f}")


if __name__ == "__main__":
    train, test = data()
    print(f"train rows={len(train)}  test rows={len(test)}")
    compare(train, test)
    quantiles(train, test)
    calibrated(train, test)
    explain(train, test)
