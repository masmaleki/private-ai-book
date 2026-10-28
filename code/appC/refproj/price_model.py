"""Ridge regression on log price (numpy only) plus empirical intervals.

p50 comes from the regression; p10/p90 come from the residuals on a
calibration slice that the regression was NOT fitted on."""

import json
import numpy as np
from .config import ARTIFACTS, CALIB_FROM, MODEL_VERSION
from .features import Featurizer, shared_rows, time_split

MODEL_FILE = ARTIFACTS / "price_model.json"


class PriceModel:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y_log):
        A = np.c_[np.ones(len(X)), X.values]
        reg = self.alpha * np.eye(A.shape[1])
        reg[0, 0] = 0  # do not shrink the intercept
        self.w = np.linalg.solve(A.T @ A + reg, A.T @ y_log)
        self.columns = list(X.columns)
        return self

    def predict_log(self, X):
        return np.c_[np.ones(len(X)), X[self.columns].values] @ self.w

    def calibrate(self, X, y_log):
        r = y_log - self.predict_log(X)
        self.q10, self.q90 = (float(v) for v in np.quantile(r, [0.1, 0.9]))
        return self

    def predict(self, X):
        m = self.predict_log(X)
        return np.exp(m + self.q10), np.exp(m), np.exp(m + self.q90)


def train(offers):
    """Fit on early train rows, calibrate intervals on the late ones."""
    tr, te = time_split(shared_rows(offers))
    fit_df = tr[tr.observed_at < CALIB_FROM]
    cal_df = tr[tr.observed_at >= CALIB_FROM]
    feat = Featurizer().fit(fit_df)
    model = PriceModel().fit(feat.transform(fit_df),
                             np.log(fit_df.unit_price.values))
    model.calibrate(feat.transform(cal_df), np.log(cal_df.unit_price.values))
    return model, feat, tr, te


def evaluate(model, feat, test_df, train_df):
    y = test_df.unit_price.values
    p10, p50, p90 = model.predict(feat.transform(test_df))
    # Baseline: median train price per part, ignoring quantity and supplier.
    base = test_df.mpn.map(train_df.groupby("mpn").unit_price.median())

    def mape(pred):
        return float(np.mean(np.abs(pred - y) / y))

    return dict(n_train=int(len(train_df)), n_test=int(len(y)),
                mape_model=round(mape(p50), 4),
                mape_baseline=round(mape(base.values), 4),
                coverage_p10_p90=round(float(np.mean((y >= p10) & (y <= p90))),
                                       3))


def save(model, feat, metrics, path=MODEL_FILE):
    path.parent.mkdir(exist_ok=True)
    card = dict(version=MODEL_VERSION, columns=model.columns,
                w=model.w.tolist(), q10=model.q10, q90=model.q90,
                features=feat.state(), metrics=metrics)
    path.write_text(json.dumps(card, indent=1))


def load(path=MODEL_FILE):
    card = json.loads(path.read_text())
    m = PriceModel()
    m.columns, m.w = card["columns"], np.array(card["w"])
    m.q10, m.q90 = card["q10"], card["q90"]
    return m, Featurizer.from_state(card["features"]), card
