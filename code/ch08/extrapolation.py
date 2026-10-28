"""Chapter 8: trees cannot extrapolate a trend (synthetic example)."""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

rng = np.random.default_rng(8)
day = np.arange(0, 730)
price = 1.00 + 0.0004 * day + rng.normal(0, 0.02, len(day))   # +0.04 cent per day
train, test = day < 365, day >= 365
X = day.reshape(-1, 1)
models = {"linear": LinearRegression(),
          "random forest": RandomForestRegressor(n_estimators=200, random_state=0,
                                                 n_jobs=1)}
for name, m in models.items():
    m.fit(X[train], price[train])
    p = m.predict(X[test])
    mae = np.abs(p - price[test]).mean()
    print(f"{name:<14} prediction day 400={p[35]:.3f}  day 729={p[-1]:.3f}  "
          f"test MAE={mae:.3f}")
print(f"true trend     day 400={1 + 0.0004 * 400:.3f}  day 729={1 + 0.0004 * 729:.3f}")
print(f"max price seen in training: {price[train].max():.3f}")
