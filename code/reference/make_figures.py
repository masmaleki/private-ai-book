"""Figures for appendix C (real outputs of the reference project)."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from refproj import db, price_model

OUT = Path(__file__).resolve().parents[2] / "assets" / "img" / "appC"
OUT.mkdir(parents=True, exist_ok=True)

model, feat, tr, te = price_model.train(db.offers_frame(db.connect()))
p10, p50, p90 = model.predict(feat.transform(te))
y = te.unit_price.values

# 1) one part over time: actual test offers vs the model's P10-P90 band
grid = pd.date_range("2026-01-01", "2026-06-30", freq="7D")
fig, ax = plt.subplots(figsize=(7, 3.3))
for q, c in ((100, "#3b7dd8"), (5000, "#e07b39")):
    rows = pd.DataFrame(dict(mpn="TPS5430DDAR", qty=q, supplier_id=1,
                             observed_at=grid))
    lo, mid, hi = model.predict(feat.transform(rows))
    ax.plot(grid, mid, color=c, label=f"P50, qty {q}")
    ax.fill_between(grid, lo, hi, color=c, alpha=.15)
    m = ((te.mpn == "TPS5430DDAR") & (te.supplier_id == 1)
         & (te.qty == q)).values
    ax.scatter(te.observed_at.values[m], y[m], s=16, color=c,
               edgecolor="k", linewidth=.4, label=f"actual, qty {q}")
ax.set_ylabel("unit price EUR")
ax.set_title("TPS5430DDAR, Supplier A: test offers vs P10-P90 (synthetic)")
ax.legend(fontsize=7, ncol=2)
fig.autofmt_xdate()
fig.tight_layout(); fig.savefig(OUT / "band.png", dpi=160); plt.close(fig)

# 2) error per part: model vs baseline (median train price per part)
base = te.mpn.map(tr.groupby("mpn").unit_price.median()).values
err = te.assign(model=np.abs(p50 - y) / y, base=np.abs(base - y) / y)
g = err.groupby("mpn")[["model", "base"]].mean().sort_values("base")
x = np.arange(len(g))
fig, ax = plt.subplots(figsize=(7, 3.4))
ax.barh(x - .2, g.base * 100, .4, label="baseline: median per part")
ax.barh(x + .2, g.model * 100, .4, label="ridge model")
ax.set_yticks(x, g.index, fontsize=7); ax.set_xlabel("MAPE %")
ax.set_xlim(0, 21); ax.legend(fontsize=8, loc="lower right")
ax.set_title("Test error per part (synthetic data)")
fig.tight_layout(); fig.savefig(OUT / "mape_parts.png", dpi=160); plt.close(fig)
print(g.round(3).to_string())
