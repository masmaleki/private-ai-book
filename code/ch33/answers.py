"""answers.py - runnable answers for chapter 32 (fictional data)."""
import agent
from agent import run, Policy
from evaluate import tasks, evaluate

# 32-2: a global retry budget across the whole run
def run_budget(task, budget, fault, seed):
    r = run(task, policy=Policy(max_retries=2), faults=fault, seed=seed)
    n_to = sum(1 for s in r["trace"] if s.get("error") == "timeout")
    return "FAILED_BUDGET" if n_to > budget else r["status"]
for b in (1, 3):
    c = sum(run_budget(t, b, 0.3, i) == "DONE" for i, t in enumerate(tasks))
    print(f"32-2 budget={b} done={c}/200")

# 32-3: expected success without retries vs observed
def n_read_calls(t):
    stock = agent.PARTS[t["mpn"]]["stock"]
    return 3 if stock < t["qty"] else 2
for p in (0.1, 0.3):
    exp0 = sum((1 - p) ** n_read_calls(t) for t in tasks) / len(tasks)
    exp2 = sum((1 - p ** 3) ** n_read_calls(t) for t in tasks) / len(tasks)
    o0 = evaluate(p, 0)["DONE"] / 200; o2 = evaluate(p, 2)["DONE"] / 200
    print(f"32-3 p={p}: expected r0={exp0:.3f} observed={o0:.3f} | "
          f"expected r2={exp2:.3f} observed={o2:.3f}")
print("32-3 share with 3 read calls:", sum(n_read_calls(t) == 3 for t in tasks) / 200)

# 32-4: idempotency key that ignores the price version returns a stale draft
agent.DRAFTS.clear()
task = {"customer": "AlpenBoard Example GmbH", "mpn": "STM32F407VGT6", "qty": 2500}
first = run(task, dry_run=False)["state"]["draft"]
orig = agent.predict_price
def new_prices(mpn, qty, faults=0.0, rng=None):        # market moved overnight
    return {"mpn": mpn, "qty": qty, "p10": 7.40, "p50": 8.10, "p90": 9.00}
agent.TOOLS["predict_price"] = new_prices
second = run(task, dry_run=False)["state"]["draft"]
print("32-4 first:", first["draft_id"], first["lines"][0]["unit_price"],
      "| rerun after price change:", second)
third = run(task | {"price_version": "pm-2026-10-18"}, dry_run=False)["state"]["draft"]
print("32-4 with price_version in key:", third["draft_id"], third["lines"][0]["unit_price"])
agent.TOOLS["predict_price"] = orig

# 32-7: PR-only policy for a coding agent
ALLOWED = {"read_file", "run_tests", "create_branch", "commit_to_branch", "open_pull_request"}
def coding_policy(action, target=""):
    if action not in ALLOWED:
        return f"DENY:{action}"
    if action == "commit_to_branch" and target in ("main", "master", "release"):
        return f"DENY:protected:{target}"
    if action == "read_file" and any(s in target for s in (".env", "secrets", "id_rsa")):
        return f"DENY:secret:{target}"
    return "ALLOW"
for a in [("read_file", "src/pricing.py"), ("commit_to_branch", "agent/fix-123"),
          ("commit_to_branch", "main"), ("merge_pull_request", "#41"),
          ("read_file", "config/.env"), ("open_pull_request", "agent/fix-123")]:
    print("32-7", a, "->", coding_policy(*a))
