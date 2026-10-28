"""agent_loop.py - the tool-calling loop: model proposes, code executes.

MockModel replays a scripted, example sequence in the OpenAI 'tool_calls'
format so the loop runs without a GPU. Swap in a real OpenAI-compatible
server (chapter 20) by replacing MockModel.chat."""

import json, textwrap
from run_demo import start, rpc

MAX_STEPS = 4  # hard budget: a loop that never ends is a bug


class MockModel:
    """Example only: decides the next step by looking at what it has seen."""

    def chat(self, messages, tools):
        seen = [m["name"] for m in messages if m["role"] == "tool"]
        if "get_inventory" not in seen:
            return {
                "tool_calls": [
                    {
                        "id": "c1",
                        "function": {
                            "name": "get_inventory",
                            "arguments": json.dumps({"mpn": "STM32F407VGT6"}),
                        },
                    },
                    {
                        "id": "c2",
                        "function": {
                            "name": "predict_price",
                            "arguments": json.dumps(
                                {"mpn": "STM32F407VGT6", "qty": 2500}
                            ),
                        },
                    },
                ]
            }
        if "get_part_offers" not in seen:
            return {
                "tool_calls": [
                    {
                        "id": "c3",
                        "function": {
                            "name": "get_part_offers",
                            "arguments": json.dumps(
                                {"mpn": "STM32F407VGT6", "min_qty": 1000}
                            ),
                        },
                    }
                ]
            }
        inv = json.loads(
            next(
                m["content"]
                for m in messages
                if m.get("name") == "get_inventory"
            )
        )
        price = json.loads(
            next(
                m["content"]
                for m in messages
                if m.get("name") == "predict_price"
            )
        )
        return {
            "content": (
                f"Stock: {inv['available']} available "
                f"(snapshot {inv['snapshot_at']}), short of 2500. "
                f"Suggested unit price P50 {price['p50']} EUR "
                f"(range {price['p10']}-{price['p90']}). "
                "One offer >=1000 pcs found; see table. "
                "Draft only - needs human review."
            )
        }


def run(question, token="tok-sarah", model=None):
    model = model or MockModel()
    srv = start(token)
    rpc(srv, 0, "initialize")
    tools = rpc(srv, 1, "tools/list")["result"]["tools"]
    messages = [
        {"role": "system", "content": "Use tools for every number."},
        {"role": "user", "content": question},
    ]
    rid = 2
    for step in range(1, MAX_STEPS + 1):
        reply = model.chat(messages, tools)
        if "tool_calls" not in reply:
            print(f"step {step}: final answer")
            print(
                textwrap.fill(
                    reply["content"],
                    72,
                    initial_indent="  ",
                    subsequent_indent="  ",
                )
            )
            break
        messages.append(
            {"role": "assistant", "tool_calls": reply["tool_calls"]}
        )
        for call in reply["tool_calls"]:
            fn = call["function"]
            res = rpc(
                srv,
                rid,
                "tools/call",
                {"name": fn["name"], "arguments": json.loads(fn["arguments"])},
            )["result"]
            rid += 1
            print(
                f"step {step}: {fn['name']} -> "
                f"{'ERROR' if res['isError'] else 'ok'}"
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": fn["name"],
                    "content": res["content"][0]["text"],
                }
            )
    else:
        print("stopped: step budget exhausted")
    srv.stdin.close()
    srv.wait()
    return messages


if __name__ == "__main__":
    run("Do we have 2500 STM32F407VGT6 and what unit price should we quote?")
