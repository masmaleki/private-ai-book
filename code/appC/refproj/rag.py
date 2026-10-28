"""Mini RAG: chunk -> scan -> BM25 index -> ACL filter -> top-k with ids.

Pure Python so it runs anywhere. In production, swap the index for
Elasticsearch/OpenSearch or a vector store; keep the ACL and scan steps."""

import math
import re
from collections import Counter
from .config import DOCS

TOKEN = re.compile(r"[a-z0-9][a-z0-9.\-]*")
INJECTION = re.compile(
    r"ignore (all |any )?(previous|prior) instructions|system prompt|"
    r"you are now|approve every", re.I)


def tokens(text):
    return [t.strip(".-") for t in TOKEN.findall(text.lower())]


def parse(path):
    head, body = path.read_text().split("---\n", 2)[1:]
    meta = dict(line.split(": ", 1) for line in head.strip().splitlines())
    meta["acl"] = set(meta["acl"].split(","))
    return meta, body.strip()


class Index:
    def __init__(self, k1=1.2, b=0.75):
        self.k1, self.b = k1, b
        self.chunks, self.quarantine = [], []

    def add_folder(self, folder=DOCS):
        for path in sorted(folder.glob("*.md")):
            meta, body = parse(path)
            for i, para in enumerate(body.split("\n\n")):
                cid = f"{meta['doc_id']}#{i}"
                if INJECTION.search(para):  # keep it out, but keep a record
                    self.quarantine.append(cid)
                    continue
                self.chunks.append(dict(id=cid, meta=meta, text=para,
                                        tf=Counter(tokens(para))))
        n = len(self.chunks)
        df = Counter(t for c in self.chunks for t in c["tf"])
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5))
                    for t, d in df.items()}
        self.avg_len = sum(sum(c["tf"].values()) for c in self.chunks) / n
        return self

    def allowed(self, chunk, session):
        m = chunk["meta"]
        return (m["tenant"] in ("all", session["tenant"])
                and session["role"] in m["acl"])

    def score(self, q_tokens, c):
        length = sum(c["tf"].values())
        s = 0.0
        for t in q_tokens:
            f = c["tf"].get(t, 0)
            if f:
                norm = self.k1 * (1 - self.b + self.b * length / self.avg_len)
                s += self.idf[t] * f * (self.k1 + 1) / (f + norm)
        return s

    def search(self, query, session, k=3):
        q = tokens(query)
        # ACL filter BEFORE ranking: forbidden text is never even scored.
        visible = [c for c in self.chunks if self.allowed(c, session)]
        hits = [(self.score(q, c), c) for c in visible]
        hits = sorted((h for h in hits if h[0] > 0), key=lambda h: -h[0])
        return [dict(id=c["id"], title=c["meta"]["title"], text=c["text"],
                     score=round(s, 3)) for s, c in hits[:k]]
