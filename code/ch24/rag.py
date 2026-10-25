"""rag.py - a small, dependency-light RAG toolkit (stdlib + numpy).

Pieces: document loading with metadata, three chunkers, a tokenizer that
keeps MPNs intact, BM25, a hashed character n-gram vectorizer (stand-in for a
neural embedding model), Reciprocal Rank Fusion and a tiny index.
"""

import math, re, zlib, pathlib, collections
from dataclasses import dataclass, field
import numpy as np


# ---------- documents ----------
@dataclass
class Doc:
    meta: dict
    text: str


def load_docs(folder="docs"):
    docs = []
    for p in sorted(pathlib.Path(folder).glob("*.md")):
        raw = p.read_text(encoding="utf-8")
        _, head, body = raw.split("---\n", 2)  # simple front matter
        meta = dict(line.split(": ", 1) for line in head.strip().splitlines())
        meta["roles"] = [r for r in meta.get("roles", "").split(",") if r]
        docs.append(Doc(meta, body.strip()))
    return docs


@dataclass
class Chunk:
    chunk_id: str
    text: str
    meta: dict = field(default_factory=dict)


# ---------- chunkers ----------
def chunk_fixed(doc, size=50, overlap=10):
    """Fixed windows of `size` words; ignores structure."""
    words, out, i = doc.text.split(), [], 0
    while i < len(words):
        out.append(" ".join(words[i : i + size]))
        i += size - overlap
    return out


def chunk_paragraph(doc, max_words=80):
    """Split on blank lines, then merge small paragraphs up to max_words."""
    out, cur = [], []
    for para in re.split(r"\n\s*\n", doc.text):
        if cur and len(" ".join(cur + [para]).split()) > max_words:
            out.append("\n\n".join(cur))
            cur = []
        cur.append(para.strip())
    if cur:
        out.append("\n\n".join(cur))
    return out


def chunk_structural(doc, max_words=120):
    """One chunk per section (## heading); tables stay whole; every chunk is
    prefixed with the document title and section so it is self-explaining."""
    title = doc.meta.get("title", "")
    sections = re.split(r"\n(?=## )", doc.text)
    out = []
    for sec in sections:
        lines = sec.strip().splitlines()
        head = lines[0].lstrip("# ").strip() if lines else ""
        body = (
            "\n".join(lines[1:]).strip() if lines[0].startswith("#") else sec
        )
        if not body:
            continue
        parts = (
            [body]
            if len(body.split()) <= max_words
            else re.split(r"\n\s*\n", body)
        )  # split long sections
        for part in parts:
            out.append(f"[{title} | {head}]\n{part.strip()}")
    return out


CHUNKERS = {
    "fixed": chunk_fixed,
    "paragraph": chunk_paragraph,
    "structural": chunk_structural,
}


def make_chunks(docs, strategy="structural"):
    chunks = []
    for d in docs:
        for i, t in enumerate(CHUNKERS[strategy](d)):
            chunks.append(Chunk(f"{d.meta['doc_id']}#{i}", t, dict(d.meta)))
    return chunks


# ---------- tokenizer ----------
WORD = re.compile(r"[0-9a-zäöüß]+(?:[-.,/][0-9a-zäöüß]+)*")
STOP = {
    "the",
    "a",
    "an",
    "of",
    "to",
    "and",
    "or",
    "is",
    "at",
    "in",
    "for",
    "with",
    "on",
    "by",
    "be",
    "from",
    "what",
    "how",
    "much",
    "der",
    "die",
    "das",
    "und",
    "ist",
    "von",
    "bei",
    "wie",
}


def tokenize(text):
    """Lowercase words; MPN-like tokens are kept whole AND split AND joined,
    so 'EXS-BUCK36-SO8', 'exs buck36' and 'EXSBUCK36SO8' can all match."""
    out = []
    for w in WORD.findall(text.lower()):
        if w in STOP:
            continue
        out.append(w)
        if "-" in w or "/" in w:
            parts = re.split(r"[-/]", w)
            out.extend(p for p in parts if p)
            out.append("".join(parts))
    return out


# ---------- BM25 ----------
class BM25:
    def __init__(self, texts, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = [collections.Counter(tokenize(t)) for t in texts]
        self.len = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.len) / len(self.len)
        df = collections.Counter(t for d in self.docs for t in d)
        n = len(self.docs)
        self.idf = {
            t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()
        }

    def score(self, query):
        q = tokenize(query)
        scores = []
        for d, dl in zip(self.docs, self.len):
            s = 0.0
            for t in q:
                tf = d.get(t, 0)
                if tf:
                    s += (
                        self.idf[t]
                        * tf
                        * (self.k1 + 1)
                        / (
                            tf
                            + self.k1 * (1 - self.b + self.b * dl / self.avg)
                        )
                    )
            scores.append(s)
        return np.array(scores)


# ---------- hashed character n-gram vectors ----------
class NgramVectorizer:
    """Stand-in for a local embedding model: hashed char 3-5 grams, log TF,
    IDF weighting, L2-normalised. Robust to spelling variants, NOT semantic."""

    def __init__(self, texts, dim=4096, n=(3, 5)):
        self.dim, self.n = dim, n
        raw = np.stack([self._tf(t) for t in texts])
        df = (raw > 0).sum(axis=0)
        self.idf = np.log((1 + len(texts)) / (1 + df)) + 1
        self.matrix = self._norm(raw * self.idf)

    def _tf(self, text):
        v = np.zeros(self.dim)
        s = " " + re.sub(r"[^0-9a-zäöüß]+", " ", text.lower()) + " "
        for n in range(self.n[0], self.n[1] + 1):
            for i in range(len(s) - n + 1):
                v[zlib.crc32(s[i : i + n].encode()) % self.dim] += 1
        return np.log1p(v)

    @staticmethod
    def _norm(m):
        nrm = np.linalg.norm(m, axis=-1, keepdims=True)
        return m / np.where(nrm == 0, 1, nrm)

    def score(self, query):
        return self.matrix @ self._norm(self._tf(query) * self.idf)


# ---------- fusion and index ----------
def rrf(rankings, k=60):
    """Reciprocal Rank Fusion: sum of 1/(k + rank) over all rankings."""
    fused = collections.defaultdict(float)
    for ranking in rankings:
        for rank, idx in enumerate(ranking, start=1):
            fused[idx] += 1.0 / (k + rank)
    return sorted(fused, key=fused.get, reverse=True), fused


class Index:
    def __init__(self, chunks):
        self.chunks = chunks
        texts = [c.text for c in chunks]
        self.bm25 = BM25(texts)
        self.vec = NgramVectorizer(texts)

    def search(self, query, k=5, mode="hybrid", allowed=None):
        """allowed: optional set of chunk indexes the caller may see
        (chapter 24 builds it from ACLs BEFORE ranking)."""
        idx = (
            list(range(len(self.chunks)))
            if allowed is None
            else sorted(allowed)
        )
        b = self.bm25.score(query)
        v = self.vec.score(query)
        rank_b = sorted((i for i in idx if b[i] > 0), key=lambda i: -b[i])
        rank_v = sorted(idx, key=lambda i: -v[i])
        if mode == "bm25":
            order, sc = rank_b, {i: b[i] for i in idx}
        elif mode == "vector":
            order, sc = rank_v, {i: v[i] for i in idx}
        else:
            order, sc = rrf([rank_b[:50], rank_v[:50]])
        return [(self.chunks[i], float(sc[i])) for i in order[:k]]
