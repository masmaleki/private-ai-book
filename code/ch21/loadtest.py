"""loadtest.py - simple concurrency load test for an OpenAI-compatible endpoint.
Measures end-to-end latency percentiles and output tokens/s at several
concurrency levels. Use synthetic prompts only (no customer data).
"""
import os, statistics, sys, time
from concurrent.futures import ThreadPoolExecutor
from client import chat

PROMPT = ("Extract MPN and quantity as JSON from this example BOM line: "
          "'C12, 100nF 50V X7R 0603, GRM188R71H104KA93D, qty 400'. ") * 10

RESULTS = []

def one_request(_):
    t0 = time.perf_counter()
    out = chat([{"role": "user", "content": PROMPT}], max_tokens=128)
    return time.perf_counter() - t0, out["usage"]["completion_tokens"]

def run(concurrency, n_requests):
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        results = list(ex.map(one_request, range(n_requests)))
    wall = time.perf_counter() - t0
    lat = sorted(r[0] for r in results)
    toks = sum(r[1] for r in results)
    p95 = lat[max(0, int(0.95 * len(lat)) - 1)]
    RESULTS.append({"conc": concurrency, "p50": statistics.median(lat),
                    "p95": p95, "tps": toks / wall})
    print(f"conc={concurrency:>3} reqs={n_requests:>3} "
          f"p50={statistics.median(lat):5.2f}s p95={p95:5.2f}s "
          f"throughput={toks / wall:7.1f} tok/s")

if __name__ == "__main__":
    levels = [int(x) for x in sys.argv[1:]] or [1, 4, 8, 16]
    for c in levels:
        run(c, n_requests=max(8, 2 * c))
    if os.environ.get("LOADTEST_JSON"):
        import json
        with open(os.environ["LOADTEST_JSON"], "w") as f:
            json.dump(RESULTS, f, indent=1)
