"""Chapter 8: early stopping on a temporal validation window (synthetic data)."""
import lightgbm as lgb
import numpy as np
import pandas as pd

from models import CUTOFF, FEATS, data, mape

train, test = data()
val_start = CUTOFF - pd.DateOffset(months=3)
fit, val = train[train.issued_at < val_start], train[train.issued_at >= val_start]
print(f"fit rows={len(fit)}  validation rows={len(val)}  test rows={len(test)}")
for lr in (0.3, 0.1, 0.03):
    m = lgb.LGBMRegressor(n_estimators=3000, learning_rate=lr, num_leaves=15,
                          min_child_samples=20, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.8, verbose=-1, random_state=0, n_jobs=2)
    m.fit(fit[FEATS], fit["rel"], eval_set=[(val[FEATS], val["rel"])],
          callbacks=[lgb.early_stopping(100, verbose=False)])
    best = m.best_iteration_
    v = mape(val.y, val.market + m.predict(val[FEATS], num_iteration=best))
    t = mape(test.y, test.market + m.predict(test[FEATS], num_iteration=best))
    print(f"learning_rate={lr:<5} best trees={best:>4}  val MAPE={v:.2f}%  test MAPE={t:.2f}%")
