"""run_demo.py - plays the role of the orchestrator talking to tool_server.py.
The 'model proposals' below are hand-written examples of what an LLM might
propose, including bad ones, to show how the server reacts."""

import json, os, subprocess, sys, textwrap


def start(token):
    env = dict(os.environ, SESSION_TOKEN=token, AUDIT_LOG="audit.jsonl")
    return subprocess.Popen(
        [sys.executable, "tool_server.py"],
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )


def rpc(proc, rid, method, params=None):
    proc.stdin.write(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": rid,
                "method": method,
                "params": params or {},
            }
        )
        + "\n"
    )
    proc.stdin.flush()
    return json.loads(proc.stdout.readline())


if __name__ == "__main__":
    if os.path.exists("audit.jsonl"):
        os.remove("audit.jsonl")
    p = start("tok-sarah")
    rpc(p, 1, "initialize")
    tools = rpc(p, 2, "tools/list")["result"]["tools"]
    print("tools:", [t["name"] for t in tools])
    proposals = [  # example model proposals
        ("get_inventory", {"mpn": "STM32F407VGT6"}),
        ("get_part_offers", {"mpn": "stm32f407vgt6", "min_qty": 1000}),
        ("get_part_offers", {"mpn": "STM32F407VGT6", "min_qty": 20000}),
        ("predict_price", {"mpn": "STM32F407VGT6", "qty": 2500}),
        ("get_part_offers", {"mpn": "TPS5430DDAR", "tenant_id": "nordsensor"}),
        ("create_purchase_order", {"mpn": "TPS5430DDAR", "qty": 10000}),
        ("get_inventory", {"mpn": "X"}),
    ]
    for i, (name, args) in enumerate(proposals, start=3):
        r = rpc(p, i, "tools/call", {"name": name, "arguments": args})[
            "result"
        ]
        text = r["content"][0]["text"]
        print(f"\n> {name}({json.dumps(args)})")
        print(
            textwrap.fill(
                f"isError={r['isError']} {text}",
                76,
                initial_indent="  ",
                subsequent_indent="    ",
            )
        )
    p.stdin.close()
    p.wait()
    q = start("tok-reza")  # developer role: no sales tools
    print("\nreza sees tools:", rpc(q, 1, "tools/list")["result"]["tools"])
    q.stdin.close()
    q.wait()
    print("\naudit log (user, tenant, tool, status):")
    for line in open("audit.jsonl"):
        a = json.loads(line)
        print(
            f"  {a['user']:<6} {a['tenant']:<11} {a['tool']:<22} {a['status']}"
        )
