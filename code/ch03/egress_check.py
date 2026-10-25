"""egress_check.py - Chapter 3: list outbound TCP connections and test egress.
Teaching script. Run on the AI server itself. Linux only (uses `ss`).
"""
import ipaddress
import shutil
import subprocess
import sys

# Networks we consider "inside the trust boundary" (example values, adapt!)
ALLOWED_NETS = [
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("::1/128"),
]

def is_allowed(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip.strip("[]").split("%")[0])
    except ValueError:
        return False
    if addr.version == 6 and addr.ipv4_mapped:
        addr = addr.ipv4_mapped
    return any(addr in net for net in ALLOWED_NETS)

def outbound_connections():
    """Parse `ss -tnp` (established TCP) into (local, peer, process)."""
    out = subprocess.run(["ss", "-tnp", "state", "established"],
                         capture_output=True, text=True, check=True).stdout
    rows = []
    for line in out.splitlines()[1:]:
        parts = line.split()
        if len(parts) < 4:
            continue
        local, peer = parts[2], parts[3]
        proc = parts[4] if len(parts) > 4 else "-"
        peer_ip = peer.rsplit(":", 1)[0]
        rows.append((local, peer, proc, is_allowed(peer_ip)))
    return rows

def egress_test(urls, timeout=3):
    """Try to reach public URLs. On a sealed AI server, ALL must fail."""
    results = []
    for url in urls:
        r = subprocess.run(["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
                            "--max-time", str(timeout), url],
                           capture_output=True, text=True)
        reachable = r.returncode == 0 and r.stdout not in ("", "000")
        results.append((url, reachable, r.stdout or "-", r.returncode))
    return results

if __name__ == "__main__":
    if not shutil.which("ss"):
        sys.exit("ss not found (install iproute2)")
    print("== established TCP connections ==")
    conns = outbound_connections()
    outside = [c for c in conns if not c[3]]
    print(f"total: {len(conns)}  inside: {len(conns) - len(outside)}"
          f"  OUTSIDE: {len(outside)}")
    for local, peer, proc, _ in outside:
        print(f"[OUTSIDE] {peer:<24} {proc[:44]}")

    print("\n== egress test (should all be BLOCKED on an AI server) ==")
    tests = ["https://huggingface.co", "https://pypi.org/simple/",
             "https://example.com"]
    leaks = 0
    for url, reachable, code, rc in egress_test(tests):
        leaks += reachable
        state = "REACHABLE (leak!)" if reachable else "blocked"
        print(f"{url:<28} {state:<18} http={code} curl_rc={rc}")
    print("RESULT:", "FAIL - egress is open" if leaks else "PASS - no egress")
    sys.exit(1 if leaks else 0)
