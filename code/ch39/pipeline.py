"""pipeline.py - a tiny Learning-Core cycle with gates, tracked in MLflow.

steps: load snapshot -> train challenger -> evaluate vs champion on the
SAME holdout -> gates -> register (or stop). Synthetic data, local MLflow
SQLite store (./mlflow.db), no network.
"""
import json

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

rng = np.random.default_rng(42)  # synthetic price data (example)
n = 2000
log_qty = rng.uniform(np.log(100), np.log(50000), n)
lead = rng.integers(1, 30, n)
price = 2.0 * np.exp(-0.08 * log_qty) * (1 + 0.01 * lead) \
    * rng.lognormal(0, 0.05, n)
X = np.column_stack([log_qty, lead])
cut = int(n * 0.8)  # time-ordered split: last 20% is the holdout
Xtr, Xte, ytr, yte = X[:cut], X[cut:], price[:cut], price[cut:]


def wape(y, p):
    return float(np.abs(y - p).sum() / np.abs(y).sum())


GATES = {"max_wape": 0.08, "min_rel_gain": 0.02, "max_bias": 0.02}

mlflow.set_tracking_uri("sqlite:///mlflow.db")  # local file, no server
mlflow.set_experiment("price-model-demo")
results, models = {}, {}
for name, model in [("champion_linear", LinearRegression()),
                    ("challenger_gbm",
                     GradientBoostingRegressor(random_state=42))]:
    with mlflow.start_run(run_name=name):
        model.fit(Xtr, ytr)
        p = model.predict(Xte)
        m = {"wape": wape(yte, p),
             "bias": float((p - yte).sum() / yte.sum())}
        mlflow.log_params({"model": name, "n_train": len(ytr),
                           "data_snapshot": "demo_offers@2026-10-01"})
        mlflow.log_metrics(m)
        # MLflow saves sklearn models with skops (no pickle); tree models
        # need an explicit allow-list of types we created ourselves
        trusted = (["sklearn.tree._tree.Tree"] if "gbm" in name else None)
        mlflow.sklearn.log_model(model, name="model",
                                 skops_trusted_types=trusted)
        results[name] = m
        models[name] = (model, p, mlflow.active_run().info.run_id)

champ, chall = results["champion_linear"], results["challenger_gbm"]
gain = (champ["wape"] - chall["wape"]) / champ["wape"]
checks = {
    "wape_ok": chall["wape"] <= GATES["max_wape"],
    "beats_champion": gain >= GATES["min_rel_gain"],
    "bias_ok": abs(chall["bias"]) <= GATES["max_bias"],
}
print(json.dumps(results, indent=1))
print(f"relative WAPE gain: {gain:.1%}")
print("gates:", checks)

# honesty check: bootstrap the WAPE gain on the holdout (paired resampling)
p_ch, p_new = models["champion_linear"][1], models["challenger_gbm"][1]
boot = []
for _ in range(1000):
    i = rng.integers(0, len(yte), len(yte))
    boot.append(1 - wape(yte[i], p_new[i]) / wape(yte[i], p_ch[i]))
lo, hi = np.percentile(boot, [2.5, 97.5])
print(f"bootstrap 95% CI of relative gain: {lo:.1%} .. {hi:.1%}")

if all(checks.values()):
    run_id = models["challenger_gbm"][2]
    mv = mlflow.register_model(f"runs:/{run_id}/model", "price-model")
    mlflow.MlflowClient().set_registered_model_alias(
        "price-model", "challenger", mv.version)
    print(f"DECISION: promote to canary (price-model v{mv.version}, "
          "alias=challenger)")
else:
    print("DECISION: keep champion")
