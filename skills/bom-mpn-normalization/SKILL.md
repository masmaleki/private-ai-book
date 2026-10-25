---
name: bom-mpn-normalization
description: Normalize messy Bill of Materials (BOM) part numbers, strip packaging suffixes (-TR, /REEL, #PBF), and match against the organization's product catalog using Hybrid NLP + RapidFuzz.
---
# BOM & MPN Normalization Skill

## Step-by-Step Procedure
1. **Deterministic Pre-cleaning**:
   - Uppercase ASCII characters, normalize Unicode (NFKC), and strip leading/trailing whitespace.
   - Separate packaging suffixes (`-TR`, `-T&R`, `/REEL`, `#PBF`, `NOPB`) into a dedicated `packaging_constraint` field rather than discarding them silently.
2. **Tier 1 Exact & Fuzzy Lookup**:
   - Run exact prefix match first; if confidence `< 1.0`, run `RapidFuzz` Jaro-Winkler / Levenshtein against the organization's `products` table.
3. **Tier 2 Cross-Reference & Form-Fit-Function (FFF)**:
   - Only propose an alternate MPN if voltage, tolerance, package footprint (e.g., `0603` vs `0402`), and temperature grade (`-40°C..+85°C` industrial vs commercial) are verified from datasheet specifications.
