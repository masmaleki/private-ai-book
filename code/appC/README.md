# refproj - private AI reference project (example)

Small, runnable reference project for appendix C of the book. It ties
together the four building blocks on synthetic `demo_*` data:

1. private LLM server (OpenAI-compatible client, or the built-in MockLLM)
2. search and RAG (BM25, ACL filter before ranking, injection quarantine)
3. small prediction model (ridge on log price, P10/P50/P90)
4. safe tool layer (session-bound tenant, strict args, audit, human approval)

All companies, prices and documents are fictional.

    python3 -m venv .venv && . .venv/bin/activate
    pip install -r requirements.txt
    make all        # setup -> train -> eval -> test
    make demo       # one quote draft and three approval attempts
    make serve      # HTTP API on 127.0.0.1:8100

Real LLM: set LLM_BASE_URL to your private server (host must be allowed in
refproj/config.py), e.g. LLM_BASE_URL=http://127.0.0.1:8000/v1
