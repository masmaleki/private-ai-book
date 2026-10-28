"""answers.py - runnable answers for exercises 29-4, 29-5, 29-7, 29-9."""
import re
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from pricing import win_prob, RULES, model, hist, X_of

def optimise_capped(cost_p50, cost_p90, qty, segment, fast, n_comp,
                    last_accepted=None):
    top = cost_p50 * RULES["max_markup"]
    if last_accepted is not None:
        top = min(top, last_accepted * 1.15)
    grid = np.round(np.arange(cost_p50, top, 0.01), 2)
    grid = grid[grid >= cost_p90 * (1 + RULES["min_margin_pct"] / 100)]
    pw = win_prob(segment, grid / cost_p50, fast, n_comp, qty)
    ep = pw * (grid - cost_p50) * qty
    i = int(np.argmax(ep))
    return grid[i], pw[i], ep[i]

print("29-4:", optimise_capped(7.585, 8.4882, 2500, "lead_time_focused", 1, 3,
                               last_accepted=8.90))
print("29-5 coef:", np.round(model.coef_, 3))
for s in ("lead_time_focused", "balanced", "price_focused"):
    p = win_prob(s, np.array([1.10, 1.20]), 0, 3, 2500)
    print("  ", s, np.round(p, 3), "change:", round(p[1] - p[0], 3))

MONTHS = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"
DATE_PATTERNS = [rf"\bby\s+\d{{1,2}}\.?\s+{MONTHS}\b",
                 rf"\b{MONTHS}\s+\d{{1,2}}\b",
                 r"\b\d{4}-\d{2}-\d{2}\b",
                 r"\b\d{1,2}\.\d{1,2}\.\d{2,4}\b"]
def promises_date(text):
    t = text.lower()
    return any(re.search(p, t) for p in DATE_PATTERNS)
print("29-7:", promises_date("we can deliver by 15 November"),
      promises_date("we can deliver within 38 days"))

tr = hist.iloc[:3200]
keep = (tr.segment != "price_focused") | (tr.markup < 1.10)
m2 = LogisticRegression(C=10.0, max_iter=2000).fit(X_of(tr[keep]), tr[keep].won)
d = pd.DataFrame({"segment": ["price_focused"], "markup": [1.30], "fast": [0],
                  "n_comp": [3], "log_qty": [np.log(2500)]})
print("29-9 full:", win_prob("price_focused", np.array([1.30]), 0, 3, 2500),
      "restricted:", m2.predict_proba(X_of(d))[:, 1],
      "n kept price_focused:", int(((tr.segment == "price_focused") & keep).sum()))
