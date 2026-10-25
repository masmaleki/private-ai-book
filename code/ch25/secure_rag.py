"""secure_rag.py - chapter 24 additions on top of chapter 23's toolkit:
ACL pre-filter, rule-based query rewrite, a simple reranker, injection scan,
grounded answer checks. Everything runs on CPU with stdlib + numpy."""

import re, sys, pathlib

sys.path.insert(
    0, str(pathlib.Path(__file__).resolve().parent.parent / "ch23")
)
from rag import load_docs, make_chunks, Index, tokenize  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent


def load_all():
    return load_docs(HERE.parent / "ch23" / "docs") + load_docs(
        HERE / "docs_extra"
    )


# ---------- 1) access control BEFORE retrieval ----------
def can_read(meta, session):
    vis = meta.get("visibility")
    if vis == "public":
        return True
    if vis == "internal":
        roles = meta.get("roles") or []
        return session["staff"] and (not roles or session["role"] in roles)
    if vis == "tenant":
        return meta.get("tenant") == session["tenant"]
    return False  # unknown -> deny


def allowed_set(index, session):
    return {i for i, c in enumerate(index.chunks) if can_read(c.meta, session)}


# ---------- 2) injection scan at ingestion ----------
INJECTION = re.compile(
    r"ignore (all )?(previous|prior|above) instructions|"
    r"(note|message) (for|to) (the )?(ai|assistant|llm)s?|"
    r"system prompt|you are now|disregard .{0,20}rules",
    re.I,
)


def scan_chunks(chunks):
    """Return chunk ids that look like prompt injection (quarantine them)."""
    return {c.chunk_id for c in chunks if INJECTION.search(c.text)}


# ---------- 3) query rewrite (rules; an LLM can do this too) ----------
ABBREV = {
    "vin": "input voltage",
    "iout": "output current",
    "eol": "end-of-life",
    "moq": "minimum order quantity",
    "lt": "lead time",
}


def rewrite(query):
    words = [ABBREV.get(w.lower(), w) for w in query.split()]
    return " ".join(words)


# ---------- 4) reranker (stand-in for a cross-encoder model) ----------
FILLER = {
    "does",
    "do",
    "work",
    "who",
    "still",
    "it",
    "can",
    "which",
    "we",
    "our",
    "us",
    "please",
    "tell",
    "me",
    "about",
}


def rerank(query, hits, top=4):
    """Re-score the candidates: coverage of the query's content words, exact
    MPN match in the chunk, and a header match. A real system would use a
    cross-encoder model here; the interface stays the same."""
    q = set(tokenize(query))
    mpns = {
        t
        for t in q
        if "-" in t and re.search(r"[a-z]", t) and re.search(r"\d", t)
    }
    mpn_parts = (
        set().union(*[set(tokenize(m)) for m in mpns]) if mpns else set()
    )
    content = {t for t in q - mpn_parts if len(t) > 1 and t not in FILLER}
    scored = []
    for chunk, _ in hits:
        terms = set(tokenize(chunk.text))
        coverage = len(content & terms) / max(1, len(content))
        mpn_hit = 1.0 if mpns & terms else 0.0
        header = set(tokenize(chunk.text.splitlines()[0]))
        head_hit = len(content & header) / max(1, len(content))
        scored.append(
            (0.6 * coverage + 0.25 * mpn_hit + 0.15 * head_hit, chunk)
        )
    scored.sort(key=lambda x: -x[0])
    return [(c, round(s, 3)) for s, c in scored[:top]]


MIN_SCORE = 0.5  # below this, answer "I don't know" without calling the LLM


# ---------- 5) the secure pipeline ----------
class SecureRAG:
    def __init__(self):
        self.index = Index(make_chunks(load_all(), "structural"))
        self.quarantine = scan_chunks(self.index.chunks)

    def retrieve(self, query, session, k=4, use_rerank=True):
        allowed = allowed_set(self.index, session)
        allowed -= {
            i
            for i, c in enumerate(self.index.chunks)
            if c.chunk_id in self.quarantine
        }
        q = rewrite(query)
        hits = self.index.search(q, k=12 if use_rerank else k, allowed=allowed)
        return rerank(q, hits, top=k) if use_rerank else hits


# ---------- 6) answer checks ----------
CITE = re.compile(r"\[([A-Z]+-[A-Z0-9-]+#\d+)\]")


def check_answer(answer, hits):
    """Every line needs a citation to a retrieved chunk, and every number in
    the line must appear in the cited chunk. Returns a list of problems."""
    by_id = {c.chunk_id: c.text for c, _ in hits}
    problems = []
    for line in filter(None, (l.strip() for l in answer.splitlines())):
        if line.startswith("I don't know"):
            continue
        cites = CITE.findall(line)
        if not cites:
            problems.append(f"no citation: {line[:50]}")
            continue
        for cid in cites:
            if cid not in by_id:
                problems.append(f"cites unretrieved {cid}")
        body = CITE.sub("", line)
        for num in re.findall(r"\d+(?:[.,]\d+)?", body):
            if not any(num in by_id.get(cid, "") for cid in cites):
                problems.append(f"number {num} not in cited source")
    return problems


def answerable(hits):
    return bool(hits) and hits[0][1] >= MIN_SCORE
