"""Exercise 8-6: coverage of the calibrated P10-P90 band per quantity bucket."""
import lightgbm as lgb
import numpy as np
import pandas as pd

from models import CUTOFF, FEATS, data

train, test = data()
cal_start = CUTOFF - pd.DateOffset(months=3)
fit, cal = train[train.issued_at < cal_start], train[train.issued_at >= cal_start]
pred = {}
for a in (0.1, 0.9):
    m = lgb.LGBMRegressor(objective="quantile", alpha=a, n_estimators=300,
                          learning_rate=0.03, num_leaves=15, min_child_samples=20,
                          verbose=-1, random_state=0, n_jobs=2).fit(fit[FEATS], fit["rel"])
    pred[a] = (m.predict(cal[FEATS]), m.predict(test[FEATS]))
score = np.maximum(pred[0.1][0] - cal["rel"], cal["rel"] - pred[0.9][0])
n = len(score)
k = np.quantile(score, min(1.0, np.ceil((n + 1) * 0.8) / n))
lo = test.market + pred[0.1][1] - k
hi = test.market + pred[0.9][1] + k
inside = (test.y >= lo) & (test.y <= hi)
bucket = pd.cut(test.qty, [0, 100, 1000, np.inf], labels=["<=100", "101-1000", ">1000"])
out = pd.DataFrame({"bucket": bucket, "inside": inside,
                    "width": np.exp(hi - lo) - 1})
res = out.groupby("bucket", observed=True).agg(rows=("inside", "size"),
                                               coverage=("inside", "mean"),
                                               median_width=("width", "median"))
print(res.round(3).to_string())
print(f"overall coverage: {inside.mean():.3f}")
