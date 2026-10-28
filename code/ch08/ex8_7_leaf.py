"""Exercise 8-7: effect of min_samples_leaf on a random forest (relative target)."""
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from models import FEATS, data, mape

train, test = data()
for leaf in (1, 5, 20, 80):
    m = RandomForestRegressor(n_estimators=300, min_samples_leaf=leaf, random_state=0,
                              n_jobs=1).fit(train[FEATS], train["rel"])
    tr = mape(train.y, train.market + m.predict(train[FEATS]))
    te = mape(test.y, test.market + m.predict(test[FEATS]))
    print(f"min_samples_leaf={leaf:>3}: train MAPE={tr:5.2f}%  test MAPE={te:5.2f}%")
