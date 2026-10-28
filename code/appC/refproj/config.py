"""Central settings: paths, seeds, versions. Change here, not in code."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DOCS = ROOT / "docs"
ARTIFACTS = ROOT / "artifacts"
DB_PATH = DATA / "demo.db"
AUDIT_LOG = Path(os.environ.get("AUDIT_LOG", ARTIFACTS / "audit.jsonl"))

SEED = 2026
MODEL_VERSION = "price-ridge-v1"
SPLIT_DATE = "2026-01-01"  # train before, test from this date
CALIB_FROM = "2025-10-01"  # last train months used for intervals
AS_OF = "2026-07-01"  # predictions are made for the day after the data ends

# Example tenants. shared_training=False: rows never enter shared models.
# prefer: what the customer values most when we pick an offer.
TENANTS = {
    "alpenboard": dict(name="AlpenBoard Example GmbH", shared_training=True,
                       prefer="lead_time"),
    "nordsensor": dict(name="NordSensor Example AG", shared_training=True,
                       prefer="price"),
    "tenant-d": dict(name="Tenant-D (example)", shared_training=False,
                     prefer="lead_time"),
}

# Business guardrails (example values, set by the business, not by code).
MIN_MARGIN_PCT = 12.0
QUOTE_VALID_DAYS = 14

# LLM endpoint: empty = built-in MockLLM. Only these hosts are allowed.
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "")
LLM_ALLOWED_HOSTS = {"127.0.0.1", "localhost", "llm.internal.example"}
