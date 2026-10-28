"""calibrate.py - conformalized quantile regression (CQR) to fix coverage.
Train quantile models on an earlier window, measure how much the P10-P90
band must be widened on a calibration window, then apply on test. example data.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from features import build_rows, FEATURES

offers = pd.read_csv("demo_offers.csv", parse_dates=["observed_at"])
parts = pd.read_csv("demo_parts.csv")
df = build_rows(offers, parts)
CAL_START, CUT = pd.Timestamp("2025-12-01"), pd.Timestamp("2026-03-01")
fit = df[df.target_date < CAL_START]
cal = df[(df.asof_date >= CAL_START) & (df.target_date < CUT)]
test = df[df.asof_date >= CUT]
print(f"fit={len(fit)} cal={len(cal)} test={len(test)}")

params = dict(n_estimators=400, learning_rate=0.03, num_leaves=15,
              min_child_samples=30, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, random_state=42, verbose=-1, n_jobs=2)
q = {}
for a in (0.1, 0.9):
    m = lgb.LGBMRegressor(objective="quantile", alpha=a, **params)
    m.fit(fit[FEATURES], fit.y)
    q[a] = m

def band(d):
    return q[0.1].predict(d[FEATURES]), q[0.9].predict(d[FEATURES])

lo, hi = band(cal)
# conformity score: how far outside the band (log-ratio scale), <0 if inside
E = np.maximum(lo - cal.y.values, cal.y.values - hi)
n = len(E)
qhat = np.quantile(E, min(1.0, np.ceil((n + 1) * 0.8) / n))
print(f"widen band by {qhat:+.4f} on log scale (= x{np.exp(qhat):.3f})")

lo_t, hi_t = band(test)
y = test.y.values
raw = np.mean((y >= lo_t) & (y <= hi_t))
adj = np.mean((y >= lo_t - qhat) & (y <= hi_t + qhat))
w_raw = np.median(np.exp(hi_t) - np.exp(lo_t))
w_adj = np.median(np.exp(hi_t + qhat) - np.exp(lo_t - qhat))
print(f"test coverage raw={raw:.3f} -> conformal={adj:.3f} (target 0.80)")
print(f"median width (as share of ref price) raw={w_raw:.3f} adj={w_adj:.3f}")
