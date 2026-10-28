"""Minimal MCP-style tool server over stdio (JSON-RPC 2.0), teaching version.

For production use the official MCP SDK. approve_quote is NOT exposed here:
an AI client can prepare drafts, never approve them."""

import json
import os
import sys
from . import db as dbmod
from .tools import SESSIONS, Tools


def handle(tools, s, req):
    m, rid = req.get("method"), req.get("id")
    if m == "initialize":
        res = dict(serverInfo=dict(name="refproj-tools", version="0.1"),
                   capabilities=dict(tools={}))
    elif m == "tools/list":
        res = dict(tools=[
            dict(name=n, inputSchema=schema.model_json_schema())
            for n, (_, schema, roles) in tools.REGISTRY.items()
            if s["role"] in roles])
    elif m == "tools/call":
        p = req.get("params", {})
        status, out = tools.call(s, p.get("name"), p.get("arguments", {}))
        res = dict(isError=status != "ok",
                   content=[dict(type="text", text=json.dumps(out))])
    else:
        return dict(jsonrpc="2.0", id=rid,
                    error=dict(code=-32601, message="method not found"))
    return dict(jsonrpc="2.0", id=rid, result=res)


def main():
    # Session is bound by the gateway when the connection opens,
    # never chosen by the model.
    s = SESSIONS.get(os.environ.get("SESSION_TOKEN", ""))
    if s is None:
        sys.exit("unauthenticated")
    tools = Tools(dbmod.connect())
    for line in sys.stdin:
        if line.strip():
            print(json.dumps(handle(tools, s, json.loads(line))), flush=True)


if __name__ == "__main__":
    main()
