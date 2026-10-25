"""Exercise 6-8: does a one-hot 'category' feature help the linear model?"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from train_linear import CUTOFF, FEATURES, load, report

df = load()
train, test = df[df.quote_date < CUTOFF], df[df.quote_date >= CUTOFF]
test = test.dropna(subset=["last_price"])
cols = FEATURES + ["category"]
prep = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), ["category"])],
                         remainder="passthrough")
model = make_pipeline(prep, LinearRegression()).fit(train[cols], train["log_price"])
y = test["unit_price_eur"].to_numpy()
report("linear + category", y, np.exp(model.predict(test[cols])))
