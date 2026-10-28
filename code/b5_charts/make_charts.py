"""make_charts.py - all matplotlib figures for chapters 35-42.

Labels are in English on purpose (Persian captions live in the text).
All data is synthetic / illustrative unless the chapter says otherwise.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
IMG = ROOT / "assets" / "img"
sys.path.insert(0, str(ROOT / "code" / "ch42"))
sys.path.insert(0, str(ROOT / "code" / "ch37"))
plt.rcParams.update({"figure.dpi": 150, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
C1, C2, C3, C4 = "#1f4e79", "#c0504d", "#2e8b57", "#d4a017"


def save(fig, ch, name):
    out = IMG / ch / name
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out.relative_to(ROOT))


# ---------------- ch35 ----------------
def ch35():
    ranks = [1, 2, 4, 8, 16, 32, 64]
    p = [2 * 4096 * r for r in ranks]
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    ax.bar([str(r) for r in ranks], [x / 1e3 for x in p], color=C1)
    ax.axhline(4096 * 4096 / 1e3, color=C2, ls="--")
    ax.set_yscale("log")
    ax.text(0, 4096 * 4096 / 1e3 * 1.3, "full 4096x4096 update = 16.8M",
            color=C2)
    ax.set_xlabel("LoRA rank r")
    ax.set_ylabel("trainable params (thousands, log)")
    save(fig, "ch35", "lora_params_vs_rank.png")

    # rule-of-thumb training memory for a 7B model (approximate!)
    labels = ["Full FT\n(bf16+Adam)", "LoRA\n(bf16 base)", "QLoRA\n(4-bit base)"]
    weights = [14, 14, 3.9]
    grads_opt = [14 + 56, 0.4, 0.4]   # grads bf16 + Adam fp32 (m, v, master)
    act = [6, 6, 6]                   # activations: depends on batch/length
    fig, ax = plt.subplots(figsize=(5.2, 3))
    ax.bar(labels, weights, color=C1, label="base weights")
    ax.bar(labels, grads_opt, bottom=weights, color=C2,
           label="grads + optimizer")
    ax.bar(labels, act, bottom=np.add(weights, grads_opt), color=C4,
           label="activations (varies)")
    ax.set_ylabel("GB (rough, 7B model)")
    ax.legend(frameon=False)
    save(fig, "ch35", "train_memory_7b.png")


def ch35_loss(log_path):
    import ast
    steps_tr, loss_tr, ep_ev, loss_ev = [], [], [], []
    for line in open(log_path, encoding="utf-8", errors="ignore"):
        line = line.strip()
        i = line.find("{'")
        if i < 0:
            continue
        try:
            d = ast.literal_eval(line[i:])
        except Exception:
            continue
        if "loss" in d and "eval_loss" not in d and "train_runtime" not in d:
            steps_tr.append(float(d["epoch"]))
            loss_tr.append(float(d["loss"]))
        if "eval_loss" in d:
            ep_ev.append(float(d["epoch"]))
            loss_ev.append(float(d["eval_loss"]))
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    ax.plot(steps_tr, loss_tr, "o-", color=C1, label="train loss")
    ax.plot(ep_ev, loss_ev, "s--", color=C2, label="eval loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss (completion tokens)")
    ax.legend(frameon=False)
    save(fig, "ch35", "loss_curve_real.png")


# ---------------- ch36 ----------------
def ch36():
    cats = ["extraction", "rag", "refusal", "tool", "security"]
    passed = [1, 2, 1, 1, 1]
    total = [3, 2, 1, 2, 2]
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    ax.bar(cats, total, color="#dddddd", label="cases")
    ax.bar(cats, passed, color=C3, label="passed")
    ax.set_ylabel("cases")
    ax.legend(frameon=False)
    save(fig, "ch36", "harness_results.png")

    answers = ["A1", "A2", "A3", "A4", "A5", "A6"]
    corr = [5, 4, 2, 5, 3, 1]
    faith = [5, 5, 2, 4, 5, 1]
    cite = [2, 2, 0, 2, 1, 0]
    x = np.arange(6)
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    ax.bar(x - 0.25, corr, 0.25, color=C1, label="correctness (1-5)")
    ax.bar(x, faith, 0.25, color=C3, label="faithfulness (1-5)")
    ax.bar(x + 0.25, cite, 0.25, color=C4, label="citation (0-2)")
    ax.set_xticks(x, answers)
    ax.legend(frameon=False, fontsize=7, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, 1.18))
    save(fig, "ch36", "rubric_scores.png")

    # base vs fine-tuned per slice (illustrative)
    sl = ["DE email", "EN email", "short", "long", "with typo"]
    base = [0.62, 0.71, 0.75, 0.55, 0.48]
    ft = [0.81, 0.84, 0.88, 0.66, 0.50]
    x = np.arange(len(sl))
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    ax.bar(x - 0.2, base, 0.4, color="#9fb7d0", label="base")
    ax.bar(x + 0.2, ft, 0.4, color=C1, label="fine-tuned")
    ax.set_xticks(x, sl)
    ax.set_ylim(0, 1)
    ax.set_ylabel("exact match (illustrative)")
    ax.legend(frameon=False)
    save(fig, "ch36", "slices_base_vs_ft.png")


# ---------------- ch37 ----------------
def ch37():
    from psi import psi_from_samples
    rng = np.random.default_rng(42)
    train = rng.normal(6.0, 1.0, 5000)
    live = rng.normal(6.5, 1.0, 2000)
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    ax.hist(train, bins=40, density=True, alpha=0.6, color=C1,
            label="training (reference)")
    ax.hist(live, bins=40, density=True, alpha=0.6, color=C2,
            label="last 4 weeks (live)")
    ax.set_xlabel("log(quantity), synthetic")
    ax.legend(frameon=False)
    ax.set_title(f"PSI = {psi_from_samples(train, live):.2f}", fontsize=9)
    save(fig, "ch37", "psi_hist.png")

    weeks = np.arange(1, 27)
    shift = np.where(weeks < 14, 0.03 * rng.standard_normal(26),
                     0.05 * (weeks - 13))
    vals = [psi_from_samples(train, rng.normal(6 + s, 1, 1500))
            for s in shift]
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    ax.plot(weeks, vals, "o-", color=C1, ms=3)
    ax.axhline(0.1, color=C4, ls="--")
    ax.axhline(0.25, color=C2, ls="--")
    ax.text(1, 0.11, "watch (0.10)", color=C4)
    ax.text(1, 0.26, "act (0.25)", color=C2)
    ax.set_xlabel("week (synthetic)")
    ax.set_ylabel("PSI vs training")
    save(fig, "ch37", "psi_weekly.png")

    # canary: error of champion vs challenger over days
    days = np.arange(1, 15)
    champ = 0.060 + 0.004 * rng.standard_normal(14)
    chall = 0.055 + 0.004 * rng.standard_normal(14)
    fig, ax = plt.subplots(figsize=(5.2, 2.4))
    ax.plot(days, champ, "o-", color="#9fb7d0", label="champion", ms=3)
    ax.plot(days, chall, "s-", color=C1, label="challenger (10% canary)",
            ms=3)
    ax.set_xlabel("day of canary (synthetic)")
    ax.set_ylabel("WAPE")
    ax.legend(frameon=False)
    save(fig, "ch37", "canary_wape.png")


# ---------------- ch38 ----------------
def ch38():
    threats = {"Indirect prompt injection": (4, 4),
               "Cross-tenant leak via RAG": (3, 5),
               "Tool misuse (send/approve)": (3, 5),
               "Secrets in logs": (4, 3),
               "Poisoned feedback data": (2, 4),
               "Malicious model file": (2, 5),
               "Prompt/PII in telemetry": (3, 4),
               "Model extraction via API": (2, 2),
               "DoS by huge prompts": (3, 2)}
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    for x in range(1, 6):
        for y in range(1, 6):
            s = x * y
            col = "#e8f3e8" if s < 6 else "#fff4d6" if s < 12 else "#f9dcdc"
            ax.add_patch(plt.Rectangle((x - .5, y - .5), 1, 1, color=col))
    for i, (name, (l, imp)) in enumerate(threats.items()):
        jitter = (i % 3 - 1) * 0.18
        ax.plot(l + jitter, imp, "o", color=C1)
        ax.annotate(str(i + 1), (l + jitter, imp), xytext=(3, 3),
                    textcoords="offset points", fontsize=7)
    ax.set_xlim(.5, 5.5)
    ax.set_ylim(.5, 5.5)
    ax.set_xlabel("likelihood (1-5)")
    ax.set_ylabel("impact (1-5)")
    legend = "\n".join(f"{i + 1}. {n}" for i, n in enumerate(threats))
    ax.text(5.7, 3, legend, fontsize=6.5, va="center")
    save(fig, "ch38", "risk_matrix.png")

    layers = ["input filter", "prompt design", "tool policy",
              "tenant from token", "approval gate", "output filter",
              "audit + alert"]
    caught = [20, 15, 30, 10, 15, 5, 5]  # illustrative share of attacks
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    ax.barh(layers[::-1], np.cumsum(caught)[::-1], color=C3)
    ax.set_xlabel("cumulative % of red-team attacks stopped (illustrative)")
    save(fig, "ch38", "defense_in_depth.png")


# ---------------- ch39 ----------------
def ch39():
    parts = ["model weights\n(8B, bf16)", "container\nimages",
             "python/OS\npackages", "adapters", "docs + SBOM"]
    gb = [16, 9, 4, 0.2, 0.05]  # illustrative sizes
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    ax.bar(parts, gb, color=[C1, C2, C4, C3, "#888888"])
    for i, v in enumerate(gb):
        ax.text(i, v + 0.3, f"{v:g} GB", ha="center", fontsize=7)
    ax.set_ylabel("GB (illustrative)")
    save(fig, "ch39", "bundle_composition.png")


# ---------------- ch40 ----------------
def ch40():
    rows = [("call audio", 90), ("call transcripts", 365),
            ("sales emails (RAG index)", 730), ("prompt logs (redacted)", 30),
            ("audit log", 2190), ("FT dataset samples", 730)]
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    for i, (n, d) in enumerate(rows):
        ax.barh(i, d, color=C1 if d < 400 else C4)
        ax.text(d + 20, i, f"{d} d", va="center", fontsize=7)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_xlabel("example retention (days) - set yours with your DPO")
    ax.invert_yaxis()
    save(fig, "ch40", "retention_example.png")


# ---------------- ch42 ----------------
def ch42():
    from capacity_tco import Gpu, Model, decode_tok_s_per_user, rent, tco
    m, g = Model(), Gpu()
    users = np.arange(1, 49)
    for ctx, col in [(2048, C3), (4096, C1), (8192, C2)]:
        plt.plot(users, [decode_tok_s_per_user(m, g, u, ctx) for u in users],
                 color=col, label=f"context {ctx}")
    fig = plt.gcf()
    fig.set_size_inches(5.2, 2.8)
    ax = plt.gca()
    ax.set_xlabel("concurrent sequences")
    ax.set_ylabel("tok/s per user (upper bound)")
    ax.axhline(15, color="#888888", ls=":")
    ax.text(30, 16, "comfortable reading speed?", fontsize=7)
    ax.legend(frameon=False)
    save(fig, "ch42", "toks_vs_users.png")

    on_prem, _ = tco(100, 48, 2.5, 36)
    cloud = rent(6.0, 36)
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    ax.plot(on_prem, color=C1, label="own server (capex 100 + 2.5/month)")
    ax.plot(cloud, color=C2, label="rented GPU (6/month)")
    ax.set_xlabel("month")
    ax.set_ylabel("cumulative cost (units)")
    ax.legend(frameon=False, fontsize=7)
    save(fig, "ch42", "tco_curves.png")

    # VRAM breakdown for three example setups
    setups = ["8B bf16\n48 GB GPU", "8B 4-bit\n24 GB GPU", "32B 4-bit\n48 GB GPU"]
    w = [16, 4.4, 18]
    ov = [3, 2, 3]
    total = [43.2, 21.6, 43.2]
    kv = [t - a - b for t, a, b in zip(total, w, ov)]
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    ax.bar(setups, w, color=C1, label="weights")
    ax.bar(setups, ov, bottom=w, color="#888888", label="overhead")
    ax.bar(setups, kv, bottom=np.add(w, ov), color=C3, label="KV cache budget")
    ax.set_ylabel("GB (90% of VRAM)")
    ax.legend(frameon=False, fontsize=7)
    save(fig, "ch42", "vram_breakdown.png")


# ---------------- ch43 ----------------
def ch43():
    roles = ["Data Eng", "ML Eng", "Backend", "Frontend", "DevOps", "Security"]
    skills = ["SQL", "Python", "ML", "LLM/RAG", "Infra", "Security"]
    rng = np.random.default_rng(7)
    base = np.array([[3, 2, 1, 1, 1, 1], [2, 3, 3, 2, 1, 1],
                     [2, 3, 1, 2, 1, 1], [1, 1, 0, 1, 0, 1],
                     [1, 2, 0, 1, 3, 2], [1, 2, 0, 1, 2, 3]])
    fig, ax = plt.subplots(figsize=(5, 3))
    im = ax.imshow(base, cmap="Blues", vmin=0, vmax=3)
    ax.set_xticks(range(6), skills, rotation=30)
    ax.set_yticks(range(6), roles)
    for i in range(6):
        for j in range(6):
            ax.text(j, i, base[i, j], ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="0 none ... 3 can teach")
    save(fig, "ch43", "skill_matrix.png")

    weeks = np.arange(0, 7)
    planned = np.linspace(40, 0, 7)
    actual = [40, 37, 31, 28, 18, 9, 3]
    fig, ax = plt.subplots(figsize=(5.2, 2.4))
    ax.plot(weeks, planned, "--", color="#888888", label="planned")
    ax.plot(weeks, actual, "o-", color=C1, label="actual (example)")
    ax.set_xlabel("pilot week")
    ax.set_ylabel("open tasks")
    ax.legend(frameon=False)
    save(fig, "ch43", "pilot_burndown.png")


if __name__ == "__main__":
    which = sys.argv[1:] or ["ch35", "ch36", "ch37", "ch38", "ch39", "ch40",
                             "ch42", "ch43"]
    for w in which:
        if w == "ch35_loss":
            ch35_loss(ROOT / "code" / "ch35" / "train_log.txt")
        else:
            globals()[w]()
