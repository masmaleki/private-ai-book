"""secret_scan.py - tiny secret scanner for prompts, logs and repos (teaching).

Real projects use dedicated scanners (e.g. gitleaks, trufflehog) in CI;
this shows the idea: patterns + entropy, and redaction before logging.
"""
import math
import re
import sys

PATTERNS = {
    "private_key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "bearer_token": re.compile(r"(?i)bearer\s+[a-z0-9._\-]{20,}"),
    "password_assign": re.compile(r"(?i)(password|passwd|pwd)\s*[:=]\s*\S+"),
    "db_url_with_pw": re.compile(r"\b\w+://[^/\s:]+:[^@\s]+@"),
}


def entropy(s: str) -> float:
    """Shannon entropy in bits per character."""
    if not s:
        return 0.0
    freq = {c: s.count(c) / len(s) for c in set(s)}
    return -sum(p * math.log2(p) for p in freq.values())


def scan(text: str):
    hits = [(name, m.group(0)[:24]) for name, rx in PATTERNS.items()
            for m in rx.finditer(text)]
    for tok in re.findall(r"[A-Za-z0-9+/_\-]{32,}", text):
        if entropy(tok) > 4.0:
            hits.append(("high_entropy", tok[:24]))
    return hits


def redact(text: str) -> str:
    for rx in PATTERNS.values():
        text = rx.sub("[REDACTED]", text)
    return text


if __name__ == "__main__":
    sample = (  # example data, all fake
        "user asked about TPS5430DDAR\n"
        "DB_URL=mysql://app:S3cret!@db.internal.example:3306/erp\n"
        "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.fake.signature_123\n"
        "api_key=Zq8vN2xL0pR7tY4wK9mB3cH6jD1fG5sA\n"
        "password = hunter2\n"
    )
    text = open(sys.argv[1]).read() if len(sys.argv) > 1 else sample
    for kind, frag in scan(text):
        print(f"FOUND {kind:16} {frag}...")
    print("---- redacted log line ----")
    print(redact(text))
