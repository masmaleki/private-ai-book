"""Chapter 8: when is a simpler model better? Train on small samples (synthetic)."""
import lightgbm as lgb
import numpy as np
from sklearn.linear_model import LinearRegression

from models import FEATS, data, mape

train, test = data()
recent = train.sort_values("issued_at")
print(f"{'train rows':>10} {'baseline':>9} {'linear':>8} {'lightgbm':>9}")
for n in (60, 200, 600, len(train)):
    tr = recent.tail(n)                      # most recent n quotes only
    base = mape(test.y, test.market + float(np.median(tr["rel"])))
    lin = LinearRegression().fit(tr[FEATS], tr["rel"])
    gbm = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15,
                            min_child_samples=20, verbose=-1, random_state=0,
                            n_jobs=2)
    gbm.fit(tr[FEATS], tr["rel"])
    l = mape(test.y, test.market + lin.predict(test[FEATS]))
    g = mape(test.y, test.market + gbm.predict(test[FEATS]))
    print(f"{n:>10} {base:>8.2f}% {l:>7.2f}% {g:>8.2f}%")
