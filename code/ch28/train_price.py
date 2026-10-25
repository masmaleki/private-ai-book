"""train_price.py - baselines vs LightGBM (point + quantiles), time split.
Synthetic data only (example). Run: python3 make_data.py && python3 train_price.py
"""
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
from features import build_rows, FEATURES

offers = pd.read_csv("demo_offers.csv", parse_dates=["observed_at"])
parts = pd.read_csv("demo_parts.csv")
df = build_rows(offers, parts)

# ---- time split: train on targets before cut, test strictly after ----
CUT = pd.Timestamp("2026-03-01")
train = df[df.target_date < CUT].copy()
test = df[df.asof_date >= CUT].copy()          # asof after cut -> no overlap
print(f"rows total={len(df)} train={len(train)} test={len(test)}")

# ---- baselines (in EUR) ----
test["b_last"] = test.last_price_same_qty.fillna(test.last_price_any)
test["b_med90"] = test.med90_same_qty.fillna(test.ref_price)

# ---- LightGBM on log-ratio target ----
params = dict(n_estimators=400, learning_rate=0.03, num_leaves=15,
              min_child_samples=30, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, random_state=42, verbose=-1, n_jobs=2)
models = {}
for name, obj in [("p50", dict(objective="quantile", alpha=0.5)),
                  ("p10", dict(objective="quantile", alpha=0.1)),
                  ("p90", dict(objective="quantile", alpha=0.9))]:
    m = lgb.LGBMRegressor(**params, **obj)
    m.fit(train[FEATURES], train.y)
    models[name] = m
    test[name] = test.ref_price * np.exp(m.predict(test[FEATURES]))

def metrics(y, yhat):
    err = np.abs(y - yhat)
    return {"WAPE": err.sum() / y.sum(), "MAPE": np.mean(err / y),
            "MdAPE": np.median(err / y), "bias": np.mean((yhat - y) / y)}

rows = {}
for col in ["b_last", "b_med90", "p50"]:
    rows[col] = metrics(test.y_price.values, test[col].values)
res = pd.DataFrame(rows).T
print("\n== overall test metrics (lower is better; bias ~ 0 is good) ==")
print(res.round(4).to_string())

cov = np.mean((test.y_price >= test.p10) & (test.y_price <= test.p90))
width = np.median((test.p90 - test.p10) / test.p50)
print(f"\nP10-P90 coverage = {cov:.3f} (target 0.80), "
      f"median relative width = {width:.3f}")

print("\n== MAPE by category ==")
seg = test.groupby("category", observed=True).apply(
    lambda g: pd.Series({"n": len(g),
                         "b_med90": np.mean(np.abs(g.y_price - g.b_med90) / g.y_price),
                         "lgbm": np.mean(np.abs(g.y_price - g.p50) / g.y_price)}),
    include_groups=False)
print(seg.round(4).to_string())

print("\n== MAPE by quantity bucket ==")
test["qty_bucket"] = pd.cut(test.qty, [0, 250, 1000, 100000],
                            labels=["<=250", "251-1000", ">1000"])
seg2 = test.groupby("qty_bucket", observed=True).apply(
    lambda g: pd.Series({"n": len(g),
                         "b_med90": np.mean(np.abs(g.y_price - g.b_med90) / g.y_price),
                         "lgbm": np.mean(np.abs(g.y_price - g.p50) / g.y_price)}),
    include_groups=False)
print(seg2.round(4).to_string())

imp = pd.Series(models["p50"].booster_.feature_importance("gain"),
                index=FEATURES)
print("\n== feature importance (gain share) ==")
print((imp / imp.sum()).sort_values(ascending=False).round(3).to_string())

joblib.dump({"models": models, "features": FEATURES,
             "version": "price-demo-0.1"}, "price_model.joblib")
test.to_csv("test_predictions.csv", index=False)
json.dump({k: {m: round(v, 4) for m, v in d.items()} for k, d in rows.items()}
          | {"coverage_p10_p90": round(float(cov), 4)},
          open("metrics.json", "w"), indent=1)
print("\nsaved price_model.joblib, metrics.json")
