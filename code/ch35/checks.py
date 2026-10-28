"""checks.py - deterministic checks after LLM extraction (chapter 33)."""


def check_extraction(email: str, out: dict, known_mpns: set[str]) -> list[str]:
    """Deterministic checks after the LLM; returns a list of problems."""
    problems = []
    if out.get("mpn") not in known_mpns:
        problems.append("mpn_not_in_catalog")
    if out.get("mpn", "") not in email:
        problems.append("mpn_not_in_email")       # copied wrongly?
    if str(out.get("qty")) not in email.replace(",", ""):
        problems.append("qty_not_in_email")
    return problems


if __name__ == "__main__":
    print(check_extraction("we need 500 pcs JKE656F, thanks.",
                           {"mpn": "JKE666F", "qty": 500}, {"JKE656F"}))
    # catch-rate on the real errors of probe.py
    import json
    bad = json.load(open("probe_result.json"))["bad"]
    for m, q, out in bad:
        email = f"we need {q} pcs {m}."
        print(m, check_extraction(email, json.loads(out), {m}))
