"""validate_alt_text.py - reject unsafe alternate explanations (teaching)."""
import re

BANNED = [r"fully compatible", r"100\s*%", r"guaranteed", r"identical",
          r"کاملاً سازگار", r"بدون هیچ تغییری", r"تضمین"]

def validate(text: str, level: str, cited_docs: set[str],
             allowed_docs: set[str]) -> list[str]:
    problems = []
    for pat in BANNED:
        if re.search(pat, text, flags=re.IGNORECASE):
            problems.append(f"banned phrase: {pat}")
    if "drop-in" in text.lower() and "drop-in" not in level:
        problems.append("claims drop-in but level is not drop-in")
    unknown = cited_docs - allowed_docs
    if unknown:
        problems.append(f"cites documents not in evidence: {sorted(unknown)}")
    if not cited_docs:
        problems.append("no citations at all")
    return problems

if __name__ == "__main__":
    ok_text = "Pin-compatible, fsw differs (570 vs 500 kHz) [DOC-NVL-014]"
    bad_text = "This part is a fully compatible drop-in replacement."
    allowed = {"DOC-NVL-014", "DOC-EXS-001"}
    print(validate(ok_text, "pin-compatible", {"DOC-NVL-014"}, allowed))
    print(validate(bad_text, "pin-compatible", set(), allowed))
