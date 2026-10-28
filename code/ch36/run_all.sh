#!/usr/bin/env bash
# Reproduce every number of chapter 34 (synthetic data, CPU, < 1 minute).
set -euo pipefail
python3 make_raw.py
python3 pipeline.py
python3 augment.py
python3 challenge_set.py
python3 validate.py data/extract_*.jsonl data/reply_*.jsonl
python3 contamination.py
python3 worked_example.py
python3 charts.py
