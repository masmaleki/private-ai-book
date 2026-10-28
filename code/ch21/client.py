"""client.py - minimal client for any OpenAI-compatible endpoint (stdlib only).
Works with vLLM, llama.cpp server, Ollama (/v1) or the mock server.
  LLM_BASE_URL  e.g. http://127.0.0.1:8000/v1
  LLM_API_KEY   the key configured on the server / reverse proxy
"""
import json, os, time, urllib.request

BASE = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8000/v1")
KEY = os.environ.get("LLM_API_KEY", "dev-secret")

def chat(messages, model=None, max_tokens=256, temperature=0.0, timeout=60):
    body = {"model": model or list_models()[0], "messages": messages,
            "max_tokens": max_tokens, "temperature": temperature}
    req = urllib.request.Request(
        f"{BASE}/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {KEY}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

def list_models():
    req = urllib.request.Request(f"{BASE}/models",
                                 headers={"Authorization": f"Bearer {KEY}"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return [m["id"] for m in json.loads(r.read())["data"]]

if __name__ == "__main__":
    print("models:", list_models())
    t0 = time.perf_counter()
    out = chat([
        {"role": "system", "content": "You are a sales assistant. Be brief."},
        {"role": "user",   # example request, fictional
         "content": "Summarise: customer needs 2,500 pcs "
                    "STM32F407VGT6 within 3 weeks."}],
        max_tokens=40)
    dt = time.perf_counter() - t0
    print("answer:", out["choices"][0]["message"]["content"][:70], "...")
    print("usage :", out["usage"], f"latency={dt:.2f}s")
