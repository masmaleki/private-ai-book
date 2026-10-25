# Evaluation report - price model v2 (synthetic example)

Test rows: 2000

| segment | n | WAPE base | WAPE cand | bias cand |
|---|---:|---:|---:|---:|
| ALL | 2000 | 0.142 | 0.096 | +0.021 |
| mfr=ST | 703 | 0.142 | 0.083 | +0.020 |
| mfr=TI | 679 | 0.141 | 0.082 | +0.019 |
| mfr=Murata | 409 | 0.147 | 0.083 | +0.010 |
| mfr=Microchip | 209 | 0.137 | 0.201 | +0.046 |
| qty=100 | 405 | 0.137 | 0.095 | +0.016 |
| qty=1000 | 410 | 0.140 | 0.099 | +0.027 |
| qty=20000 | 393 | 0.140 | 0.094 | +0.019 |
| tenant_A | 721 | 0.138 | 0.095 | +0.025 |
| tenant_B | 655 | 0.145 | 0.095 | +0.021 |
| tenant_C | 624 | 0.144 | 0.097 | +0.015 |

MAE(cand) - MAE(base), 95% bootstrap CI: [-0.0642, -0.0460]
P10-P90 coverage: 0.782 (target 0.80)
Pinball P90: 0.0277

Win-probability model:
ROC-AUC 0.776 | PR-AUC 0.767 | Brier 0.205 | ECE 0.098

## Release gate
- [PASS] beats baseline overall (CI upper bound < 0)
- [FAIL] no segment worse than baseline
- [PASS] interval coverage within 0.75-0.85
- [FAIL] win model ECE <= 0.05
- segments worse than baseline: mfr=Microchip

Decision: BLOCK (fix or limit scope)
