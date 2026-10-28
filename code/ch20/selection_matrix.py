"""selection_matrix.py - weighted decision matrix for model selection (example data).
Scores 1-5 come from OUR internal benchmark and licence review, not from leaderboards.
"""
criteria = {            # weight (sum = 1.0)
    "licence_ok":      0.20,
    "internal_bench":  0.30,   # BOM/RFQ extraction + RAG answers, FA/DE/EN
    "multilingual":    0.15,
    "tool_calling":    0.15,
    "vram_fit":        0.10,
    "ecosystem":       0.10,   # vLLM/llama.cpp support, quantized variants
}
candidates = {          # hypothetical candidates, example scores
    "Model-A 8B":  {"licence_ok": 5, "internal_bench": 3, "multilingual": 4,
                    "tool_calling": 4, "vram_fit": 5, "ecosystem": 5},
    "Model-B 32B": {"licence_ok": 5, "internal_bench": 5, "multilingual": 4,
                    "tool_calling": 5, "vram_fit": 3, "ecosystem": 4},
    "Model-C 24B": {"licence_ok": 2, "internal_bench": 4, "multilingual": 5,
                    "tool_calling": 4, "vram_fit": 4, "ecosystem": 4},
}
HARD_GATES = {"licence_ok": 3}   # below this score the candidate is excluded

assert abs(sum(criteria.values()) - 1.0) < 1e-9
for name, s in candidates.items():
    total = sum(criteria[c] * s[c] for c in criteria)
    gated = [c for c, minimum in HARD_GATES.items() if s[c] < minimum]
    flag = f"  EXCLUDED (gate: {', '.join(gated)})" if gated else ""
    print(f"{name:<12} weighted score = {total:.2f}{flag}")
