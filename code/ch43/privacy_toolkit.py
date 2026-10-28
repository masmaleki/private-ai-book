"""privacy_toolkit.py - small, runnable demos of privacy-enhancing techniques.

Teaching code only: synthetic data, no real people, no production crypto.
Python 3.11+, standard library + numpy.
"""
import hashlib
import hmac
import secrets
from collections import Counter

import numpy as np

# ---------- 1) pseudonymisation: plain hash vs keyed hash ----------
# example data: fictional contact e-mails
CONTACTS = [f"buyer{i:04d}@alpenboard-example.at" for i in range(10_000)]
TARGET = "buyer0042@alpenboard-example.at"


def plain_hash(email: str) -> str:
    return hashlib.sha256(email.encode()).hexdigest()


def keyed_pseudonym(email: str, key: bytes) -> str:
    # HMAC-SHA256: without the key, nobody can recompute the mapping
    return hmac.new(key, email.encode(), hashlib.sha256).hexdigest()[:16]


def dictionary_attack(leaked: str, candidates: list[str]) -> str | None:
    lookup = {plain_hash(c): c for c in candidates}  # attacker's guess list
    return lookup.get(leaked)


# ---------- 2) k-anonymity on quasi-identifiers ----------
# example data: one row per buyer contact in a fictional analytics table
ROWS = [
    {"role": "buyer", "country": "AT", "size": 120, "city": "Salzburg"},
    {"role": "buyer", "country": "AT", "size": 135, "city": "Linz"},
    {"role": "buyer", "country": "AT", "size": 140, "city": "Salzburg"},
    {"role": "buyer", "country": "DE", "size": 900, "city": "Munich"},
    {"role": "buyer", "country": "DE", "size": 950, "city": "Munich"},
    {"role": "buyer", "country": "DE", "size": 870, "city": "Stuttgart"},
    {"role": "ceo", "country": "HU", "size": 15, "city": "Budapest"},
    {"role": "buyer", "country": "HU", "size": 40, "city": "Budapest"},
]
QI = ("role", "country", "size", "city")


def k_of(rows, keys) -> int:
    groups = Counter(tuple(r[k] for k in keys) for r in rows)
    return min(groups.values())


def generalise(r: dict) -> dict:
    band = "<50" if r["size"] < 50 else "50-499" if r["size"] < 500 else "500+"
    role = "management" if r["role"] == "ceo" else "staff"
    return {"role": role, "country": r["country"], "size": band, "city": "*"}


def suppress_small_groups(rows, keys, k):
    groups = Counter(tuple(r[x] for x in keys) for r in rows)
    return [r for r in rows if groups[tuple(r[x] for x in keys)] >= k]


# ---------- 3) differential privacy: Laplace mechanism ----------
def laplace_count(true_count: int, epsilon: float, rng) -> float:
    sensitivity = 1.0  # one person changes a count by at most 1
    return true_count + rng.laplace(0.0, sensitivity / epsilon)


# ---------- 4) secure sum with additive secret sharing ----------
P = 2**61 - 1  # a large prime modulus


def share(value: int, n: int) -> list[int]:
    parts = [secrets.randbelow(P) for _ in range(n - 1)]
    parts.append((value - sum(parts)) % P)
    return parts


# ---------- 5) federated averaging (FedAvg) ----------
def fedavg(updates: list[np.ndarray], n_samples: list[int]) -> np.ndarray:
    w = np.array(n_samples, dtype=float) / sum(n_samples)
    return sum(wi * u for wi, u in zip(w, updates))


def main():
    print("== 1) pseudonymisation")
    leaked = plain_hash(TARGET)
    print("plain hash re-identified ->", dictionary_attack(leaked, CONTACTS))
    key = b"example-key-kept-in-a-secret-manager"
    print("keyed pseudonym          ->", keyed_pseudonym(TARGET, key))
    print("attack on keyed value    ->",
          dictionary_attack(keyed_pseudonym(TARGET, key), CONTACTS))

    print("== 2) k-anonymity")
    print("k on raw quasi-identifiers   :", k_of(ROWS, QI))
    gen = [generalise(r) for r in ROWS]
    print("k after generalisation       :", k_of(gen, QI))
    kept = suppress_small_groups(gen, QI, k=2)
    print("k after suppressing groups<2 :", k_of(kept, QI),
          f"(kept {len(kept)} of {len(gen)} rows)")

    print("== 3) Laplace mechanism, true count = 37")
    rng = np.random.default_rng(42)
    for eps in (1.0, 0.5, 0.1):
        draws = [laplace_count(37, eps, rng) for _ in range(10_000)]
        print(f"eps={eps:<4} one release={draws[0]:6.1f}  "
              f"std over 10k releases={np.std(draws):5.2f}")

    print("== 4) secure sum of three tenants' order counts")
    counts = {"tenant_a": 410, "tenant_b": 1250, "tenant_c": 85}
    all_shares = [share(v, 3) for v in counts.values()]
    # party j receives the j-th share from every tenant and adds them up
    partial = [sum(s[j] for s in all_shares) % P for j in range(3)]
    print("what party 0 sees from tenant_a:", all_shares[0][0] % 1000, "(random)")
    print("reconstructed total:", sum(partial) % P,
          "| true total:", sum(counts.values()))

    print("== 5) FedAvg of three local updates")
    updates = [np.array([0.20, -0.10]), np.array([0.50, 0.30]),
               np.array([-0.10, 0.00])]
    print("global update:", np.round(fedavg(updates, [200, 600, 200]), 3))


if __name__ == "__main__":
    main()
