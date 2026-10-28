"""Chapter 9 exercises 9-1, 9-2, 9-3, 9-5, 9-6 (synthetic / example data)."""
import numpy as np
import pandas as pd

from backtest import run, wape
from demand import classify
from hand_examples import croston

# 9-1: seasonal naive vs naive on six months of 2025 (example data)
ly = np.array([500, 520, 560, 380, 540, 600])         # Jun..Nov 2024
ty = np.array([530, 540, 600, 400, 580, 610])         # Jun..Nov 2025
may25 = 510
naive = np.r_[may25, ty[:-1]]
print("9-1 snaive MAE:", np.abs(ty - ly).mean().round(2),
      " naive MAE:", np.abs(ty - naive).mean().round(2))

# 9-2: Croston with alpha = 0.5
print("9-2:")
f, _ = croston(np.array([0, 200, 0, 0, 0, 100, 0, 300]), alpha=0.5)
print(f"   final Croston={f:.2f}  SBA={f * 0.75:.2f}")

# 9-3: classify a series
y = np.array([0, 0, 40, 0, 0, 0, 900, 0, 0, 60, 0, 0])
adi, cv2, label = classify(y)
print(f"9-3 ADI={adi:.2f}  CV2={cv2:.2f}  -> {label}")

r = run()
# 9-5: moving average of 6 months, computed directly from the backtest rows
from backtest import wide  # noqa: E402
Y, kind = wide()
rows = []
for T in range(35, 45):
    for h in (1, 2, 3):
        rows.append(pd.DataFrame({"kind": kind, "y": Y[:, T + h],
                                  "p": Y[:, T - 5: T + 1].mean(1)}))
ma6 = pd.concat(rows)
print("9-5 WAPE moving avg 6:")
print(" ",
      ma6.groupby("kind").apply(wape).round(1).to_dict(),
      "all:", round(wape(ma6), 1))
# 9-6: coverage by horizon
g = r[r.method == "lightgbm global"]
inside = (g.y >= g.lo) & (g.y <= g.hi)
cov = inside.groupby([g.kind, g.h]).mean().unstack()
print("9-6 coverage of [P10, P90] by kind and horizon:")
print(cov.round(3).to_string())
