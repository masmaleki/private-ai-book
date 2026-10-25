"""Quote assistant: a FIXED workflow, not a free agent.

Code decides which tools run and in which order. The LLM only writes the
explanation, and the guard checks it. The result is a draft for a human."""

from .config import TENANTS
from .llm import check_answer, facts_prompt, get_llm


def assist_quote(tools, s, customer_id, mpn, qty, margin_pct=18.0):
    status, offers = tools.call(s, "get_part_offers", {"mpn": mpn, "qty": qty})
    if status != "ok":
        return dict(status=status, detail=offers)
    if not offers:
        return dict(status="no_offers", detail=f"no offers for {mpn}")
    if TENANTS[s["tenant"]]["prefer"] == "lead_time":
        best = min(offers, key=lambda o: (o["lead_days"], o["unit_price"]))
        why = "Fastest"
    else:
        best = min(offers, key=lambda o: (o["unit_price"], o["lead_days"]))
        why = "Cheapest"
    _, pred = tools.call(s, "predict_price", {
        "mpn": mpn, "qty": qty, "supplier_id": best["supplier_id"]})
    _, docs = tools.call(s, "search_docs", {
        "query": f"{s['tenant']} lead time price quote margin", "k": 3})

    cost = best["unit_price"]
    price = round(cost / (1 - margin_pct / 100), 4)  # price from margin rule
    facts = [
        ("F1", f"{why} offer is {best['name']} at {cost} EUR "
               f"with {best['lead_days']} days lead time."),
        ("F2", f"Price model {pred['model_version']} expects {pred['p50']} "
               f"EUR, range {pred['p10']} to {pred['p90']} EUR."),
        ("F3", f"Proposed price {price} EUR gives {margin_pct} percent margin."),
    ] + [(d["id"], d["text"].replace("\n", " ")) for d in docs[:1]]

    llm = get_llm()
    answer = llm.chat(facts_prompt(f"Explain quote for {qty} x {mpn}", facts))
    problems = check_answer(answer, facts)
    if problems:  # never show an unchecked explanation
        answer = "Explanation withheld: " + "; ".join(problems)

    status, draft = tools.call(s, "create_quote_draft", {
        "mpn": mpn, "qty": qty, "supplier_id": best["supplier_id"],
        "customer_id": customer_id, "unit_price": price, "cost_price": cost,
        "explanation": answer})
    return dict(status=status, draft=draft, explanation=answer,
                facts=facts, llm=llm.name, guard_problems=problems)
