"""evaluate.py - evaluate the agent on many FICTIONAL tasks with injected
tool faults. Measures success, policy blocks, steps and retries."""
import json, random
import agent
from agent import run, Policy

rng = random.Random(32)
customers = ["AlpenBoard Example GmbH", "NordSensor Example AG", "Danube Components Example"]
tasks = [{"customer": rng.choice(customers),
          "mpn": rng.choice(list(agent.PARTS)),
          "qty": rng.choice([500, 1000, 2500, 5000, 10000])} for _ in range(200)]

def evaluate(fault, retries):
    agent.DRAFTS.clear()
    out = {"DONE": 0, "FAILED": 0, "BLOCKED": 0, "STEP_LIMIT": 0, "steps": [], "timeouts": 0}
    for i, t in enumerate(tasks):
        r = run(t, policy=Policy(max_retries=retries), faults=fault, seed=i)
        out[r["status"]] += 1
        out["steps"].append(len(r["trace"]))
        out["timeouts"] += sum(1 for s in r["trace"] if s.get("error") == "timeout")
        # invariant checks: no forbidden tool ever executed, drafts never auto-approved
        assert all(s.get("tool") not in agent.FORBIDDEN or "blocked" in s for s in r["trace"])
        if "draft" in r["state"]:
            assert r["state"]["draft"]["state"] in ("PENDING_APPROVAL", "NEEDS_SENIOR_APPROVAL")
    return out

if __name__ == "__main__":
    res = {}
    for retries in (0, 2):
        for fault in (0.0, 0.05, 0.1, 0.2, 0.3):
            o = evaluate(fault, retries)
            res[f"{retries}_{fault}"] = {k: v for k, v in o.items() if k != "steps"} | {
                "mean_steps": round(sum(o["steps"]) / len(o["steps"]), 2)}
            print(f"retries={retries} fault={fault:.2f} done={o['DONE']}/200 "
                  f"failed={o['FAILED']} timeouts={o['timeouts']} "
                  f"mean_steps={sum(o['steps'])/len(o['steps']):.2f}")
    json.dump(res, open("eval_results.json", "w"), indent=1)
    print("invariants held on all runs")
