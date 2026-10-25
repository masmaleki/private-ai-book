"""demo_secure.py - the chapter 23 leak, the injection doc, and the fixes."""

from secure_rag import SecureRAG, check_answer, answerable

rag = SecureRAG()
SARAH_AB = {
    "user": "sarah",
    "tenant": "alpenboard",
    "role": "sales",
    "staff": True,
}
print("quarantined at ingestion:", sorted(rag.quarantine))


def show(label, hits):
    print(f"  {label:<9}", ", ".join(f"{c.chunk_id}" for c, _ in hits))


q = "What does Supplier B charge for EXS-BUCK36-SO8?"
print(f"\nQ: {q}")
show("no ACL", rag.index.search(q, k=4))  # chapter 23 behaviour
hits = rag.retrieve(q, SARAH_AB)
show("secure", hits)
print(f"  top score {hits[0][1]} -> answerable={answerable(hits)}")

q2 = "Is EXS-BUCK36-SO8 end-of-life? Who still sells it?"
print(f"\nQ: {q2}")
show("no scan", rag.index.search(q2, k=4))
hits2 = rag.retrieve(q2, SARAH_AB)
show("secure", hits2)
print(f"  top score {hits2[0][1]} -> answerable={answerable(hits2)}")

q3 = "EXS-BUCK36-SO8 VIN max and IOUT at 105 °C"
hits3 = rag.retrieve(q3, SARAH_AB)
print(f"\nQ: {q3}")
print("  rewritten:", __import__("secure_rag").rewrite(q3))
print("  reranked :", [(c.chunk_id, s) for c, s in hits3])
good = (
    "Input voltage 4.5 V to 36 V [DS-0001#2]\n"
    "Max output current at 105 °C is 2.1 A [DS-0001#2]"
)
bad = (
    "Input voltage up to 40 V [DS-0001#2]\n"
    "It is end-of-life, buy from Broker C"
)
print("  check good answer:", check_answer(good, hits3))
print("  check bad answer :", check_answer(bad, hits3))
