"""egress_check_json.py - Exercise 3-5 answer: allow-list from file + JSON report."""
import ipaddress
import json
import sys
from pathlib import Path

import egress_check as ec

def load_allowed(path="allowed_nets.txt"):
    nets = []
    for line in Path(path).read_text().splitlines():
        line = line.split("#")[0].strip()      # allow comments
        if line:
            nets.append(ipaddress.ip_network(line))
    return nets

def report_json(conns, egress):
    return json.dumps({
        "outside": [{"peer": p, "process": pr} for _, p, pr, ok in conns
                    if not ok],
        "egress": [{"url": u, "reachable": r} for u, r, _, _ in egress],
    }, indent=2)

if __name__ == "__main__":
    ec.ALLOWED_NETS[:] = load_allowed(sys.argv[1] if len(sys.argv) > 1
                                      else "allowed_nets.txt")
    conns = ec.outbound_connections()
    egress = ec.egress_test(["https://example.com"])
    print(report_json(conns, egress))
