"""Chapter 6: first models, by hand and with scikit-learn (synthetic data only)."""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

HERE = Path(__file__).parent
CUTOFF = pd.Timestamp("2025-07-01")  # train before, test from this date on
FEATURES = ["log_ref", "log10_qty", "days"]


def section(title):
    print(f"\n== {title} ==")


def gd_five_points():
    """Fit price = a + b*log10(qty) on 5 example points with gradient descent."""
    x = np.log10([1, 10, 100, 1000, 10000])
    y = np.array([1.20, 1.02, 0.86, 0.70, 0.51])
    a, b, lr = 0.0, 0.0, 0.05
    for step in range(1, 2001):
        err = (a + b * x) - y                 # prediction minus truth
        grad_a = 2 * err.mean()               # d MSE / d a
        grad_b = 2 * (err * x).mean()         # d MSE / d b
        a, b = a - lr * grad_a, b - lr * grad_b
        if step in (1, 2, 10, 100, 500, 2000):
            mse = ((a + b * x - y) ** 2).mean()
            print(f"step {step:4d}: a={a:.4f} b={b:.4f} mse={mse:.6f}")
    # closed form (least squares) for comparison
    b_ls = ((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum()
    a_ls = y.mean() - b_ls * x.mean()
    print(f"least squares: a={a_ls:.4f} b={b_ls:.4f}")
    print(f"predict qty=500: {a_ls + b_ls * np.log10(500):.3f} EUR")


def load():
    q = pd.read_csv(HERE / "data/quotes.csv", parse_dates=["quote_date"])
    p = pd.read_csv(HERE / "data/parts.csv")
    df = q.merge(p, on="part_id", how="left").sort_values("quote_date")
    df["log_ref"] = np.log(df["ref_price"])
    df["log10_qty"] = np.log10(df["quantity"])
    df["days"] = (df["quote_date"] - pd.Timestamp("2024-01-01")).dt.days
    df["log_price"] = np.log(df["unit_price_eur"])
    # baseline 1: last price seen for the same part BEFORE this quote
    df["last_price"] = df.groupby("part_id")["unit_price_eur"].shift(1)
    return df.reset_index(drop=True)


def mape(y, yhat):
    return float(np.mean(np.abs(yhat - y) / y) * 100)


def report(name, y, yhat):
    print(f"{name:<22} MAE={mean_absolute_error(y, yhat):.4f} EUR  "
          f"MAPE={mape(y, yhat):5.1f}%")


def temporal_eval(df):
    train, test = df[df.quote_date < CUTOFF], df[df.quote_date >= CUTOFF]
    test = test.dropna(subset=["last_price"])
    print(f"train rows={len(train)}  test rows={len(test)}")
    y = test["unit_price_eur"].to_numpy()
    report("baseline last price", y, test["last_price"].to_numpy())
    med = train.groupby("part_id")["unit_price_eur"].median()
    report("baseline part median", y, test["part_id"].map(med).to_numpy())
    model = LinearRegression().fit(train[FEATURES], train["log_price"])
    pred = np.exp(model.predict(test[FEATURES]))
    report("linear (log space)", y, pred)
    ratio = pred / y
    print(f"model below actual in {np.mean(ratio < 1):.0%} of test rows, "
          f"median pred/actual = {np.median(ratio):.3f}")
    coefs = {f: round(float(c), 6) for f, c in zip(FEATURES, model.coef_)}
    print("coefficients:", coefs, " intercept:", round(float(model.intercept_), 4))
    return model, train


def random_split_eval(df):
    tr, te = train_test_split(df, test_size=0.25, random_state=0)
    m = LinearRegression().fit(tr[FEATURES], tr["log_price"])
    report("linear, RANDOM split", te["unit_price_eur"].to_numpy(),
           np.exp(m.predict(te[FEATURES])))


def overfit_demo():
    rng = np.random.default_rng(1)
    x = np.sort(rng.uniform(0, 4, 10))
    y = 1.2 - 0.17 * x + rng.normal(0, 0.04, 10)
    xt = np.linspace(0.1, 3.9, 50)
    yt = 1.2 - 0.17 * xt + rng.normal(0, 0.04, 50)
    for deg in (1, 3, 9):
        m = make_pipeline(PolynomialFeatures(deg), LinearRegression())
        m.fit(x[:, None], y)
        tr = mean_absolute_error(y, m.predict(x[:, None]))
        te = mean_absolute_error(yt, m.predict(xt[:, None]))
        print(f"degree {deg}: train MAE={tr:.4f}  test MAE={te:.4f}")


def save(model, train):
    out = HERE / "model"
    out.mkdir(exist_ok=True)
    path = out / "price_linear_v1.joblib"
    joblib.dump(model, path)
    meta = {"model": "price_linear", "version": "v1", "features": FEATURES,
            "target": "log(unit_price_eur)", "train_until": str(CUTOFF.date()),
            "train_rows": int(len(train)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (out / "price_linear_v1.json").write_text(json.dumps(meta, indent=2))
    print("saved", path.name, "sha256", meta["sha256"][:16], "...")


if __name__ == "__main__":
    section("1) gradient descent on 5 points")
    gd_five_points()
    data = load()
    section("2) temporal split: baselines vs linear model")
    lin, tr = temporal_eval(data)
    section("3) the same model with a random split (too optimistic)")
    random_split_eval(data)
    section("4) under- and overfitting with polynomials")
    overfit_demo()
    section("5) save model + metadata")
    save(lin, tr)
