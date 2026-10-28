"""make_figures.py - figures for chapter 22 (English labels, example data)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, os
OUT = "../../assets/img/ch22"; os.makedirs(OUT, exist_ok=True)

# 1) risk map of candidate tools: reversibility vs blast radius (illustrative)
tools = {"get_part_offers": (1, 1), "get_inventory": (1, 1.3), "predict_price": (1.4, 0.6),
         "create_quote_draft": (2.2, 2), "send_quote_email": (3.6, 3.2),
         "create_purchase_order": (4.4, 4.5), "run_sql": (4.7, 4.8)}
fig, ax = plt.subplots(figsize=(7, 4.6))
ax.axvspan(0, 2, color="#d9f2d9"); ax.axvspan(2, 3.2, color="#fff2cc")
ax.axvspan(3.2, 5, color="#f8d7da")
for n, (x, y) in tools.items():
    ax.scatter(x, y, s=60, color="k"); ax.annotate(n, (x, y), xytext=(5, 5),
                                                   textcoords="offset points", fontsize=8)
ax.text(0.2, 4.6, "auto (read-only)", fontsize=9); ax.text(2.1, 4.6, "human confirms", fontsize=9)
ax.text(3.3, 0.3, "not exposed to LLM", fontsize=9)
ax.set_xlim(0, 5); ax.set_ylim(0, 5)
ax.set_xlabel("irreversibility / side effects ->"); ax.set_ylabel("blast radius ->")
ax.set_title("Which tools may an LLM call? (illustrative)")
fig.tight_layout(); fig.savefig(f"{OUT}/tool_risk_map.png", dpi=160); plt.close(fig)

# 2) audit statuses from run_demo.py (real counts from the demo run)
st = {"ok": 4, "invalid_args": 2, "denied_unknown_tool": 1, "denied_role": 0}
fig, ax = plt.subplots(figsize=(6.5, 3.2))
ax.barh(list(st), list(st.values()), color=["#4c9a2a", "#e0a100", "#c0392b", "#7f8c8d"])
for i, v in enumerate(st.values()):
    ax.text(v + 0.05, i, str(v), va="center")
ax.set_xlabel("calls"); ax.set_title("Audit log of the demo session (run_demo.py)")
fig.tight_layout(); fig.savefig(f"{OUT}/audit_status.png", dpi=160); plt.close(fig)

# 3) timeline of a 3-step tool loop (example latencies, seconds)
steps = [("LLM: plan (2 calls)", 0.0, 0.9, "#3b7dd8"), ("get_inventory", 0.9, 0.05, "#4c9a2a"),
         ("predict_price", 0.9, 0.12, "#4c9a2a"), ("LLM: next step", 1.02, 0.7, "#3b7dd8"),
         ("get_part_offers", 1.72, 0.06, "#4c9a2a"), ("LLM: final answer", 1.78, 1.6, "#3b7dd8")]
fig, ax = plt.subplots(figsize=(7, 3.2))
for i, (n, s, d, c) in enumerate(steps):
    ax.barh(i, d, left=s, color=c); ax.text(s + d + 0.03, i, n, va="center", fontsize=8)
ax.set_yticks([]); ax.invert_yaxis(); ax.set_xlim(0, 4.3)
ax.set_xlabel("seconds (example)"); ax.set_title("Where the time goes: model calls dominate")
fig.tight_layout(); fig.savefig(f"{OUT}/loop_timeline.png", dpi=160); plt.close(fig)
print("ok")
