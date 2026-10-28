"""status_report.py - turn tracking.yaml into an honest weekly status.

Rules enforced (teaching):
  * a task may only be "done" if it has evidence
  * a task may only be "blocked" if it names what blocks it
"""
import sys
from collections import Counter
from pathlib import Path

import yaml

data = yaml.safe_load(Path(sys.argv[1] if len(sys.argv) > 1
                           else "tracking.yaml").read_text())
errors = []
for t in data["tasks"]:
    if t["status"] == "done" and not t.get("evidence"):
        errors.append(f"{t['id']}: 'done' without evidence")
    if t["status"] == "blocked" and not t.get("blocked_by"):
        errors.append(f"{t['id']}: 'blocked' without prerequisite")
if errors:
    print("TRACKING INVALID:")
    for e in errors:
        print("  -", e)
    sys.exit(1)

c = Counter(t["status"] for t in data["tasks"])
print(f"Pilot {data['pilot']} - week {data['week']}")
print("  " + "  ".join(f"{s}: {c.get(s, 0)}"
                       for s in ("done", "doing", "todo", "blocked")))
for t in data["tasks"]:
    if t["status"] == "blocked":
        print(f"  BLOCKED {t['id']} ({t['owner']}): {t['blocked_by']}")
