"""demo_chunking.py - chunk the fictional datasheet three ways."""

from rag import load_docs, CHUNKERS

doc = [d for d in load_docs() if d.meta["doc_id"] == "DS-0001"][0]
for name, fn in CHUNKERS.items():
    chunks = fn(doc)
    sizes = [len(c.split()) for c in chunks]
    # where does the 105 degC row of the table end up, and is its header there?
    hit = [i for i, c in enumerate(chunks) if "TA = 105" in c]
    with_header = [i for i in hit if "| Parameter |" in chunks[i]]
    print(
        f"{name:<10} n={len(chunks)} words={sizes} "
        f"row105={hit} header={with_header}"
    )
print("\n--- fixed chunk containing the 105 degC row ---")
c = CHUNKERS["fixed"](doc)
print(c[[i for i, x in enumerate(c) if "TA = 105" in x][0]][:300])
print("\n--- structural chunk containing the 105 degC row ---")
c = CHUNKERS["structural"](doc)
print(c[[i for i, x in enumerate(c) if "TA = 105" in x][0]])
