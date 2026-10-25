"""mock_llm_server.py - a tiny OpenAI-compatible FAKE LLM server (stdlib only).
For practising clients, auth, healthchecks and load tests WITHOUT a GPU.
It does not run a model: it echoes a canned answer and sleeps to imitate
prefill + decode time. Numbers measured against it say nothing about real GPUs.
"""
import json, os, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

API_KEY = os.environ.get("MOCK_API_KEY", "dev-secret")
MODEL = "mock-8b-instruct"
PREFILL_S_PER_1K = 0.05     # pretend: 50 ms per 1k prompt tokens
DECODE_S_PER_TOKEN = 0.01   # pretend: 100 tokens/s per request
MAX_PARALLEL = 8            # pretend batch capacity; more requests queue
slots = threading.Semaphore(MAX_PARALLEL)
stats = {"requests": 0, "tokens_out": 0}
lock = threading.Lock()

class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _authorized(self):
        return self.headers.get("Authorization", "") == f"Bearer {API_KEY}"

    def log_message(self, *args):        # keep the console quiet
        pass

    def do_GET(self):
        if self.path == "/health":
            return self._send(200, b"", "text/plain")
        if self.path == "/metrics":      # Prometheus text format
            with lock:
                txt = (f"mock_requests_total {stats['requests']}\n"
                       f"mock_generation_tokens_total {stats['tokens_out']}\n")
            return self._send(200, txt.encode(), "text/plain")
        if self.path == "/v1/models":
            if not self._authorized():
                return self._send(401, {"error": "unauthorized"})
            return self._send(200, {"object": "list",
                                    "data": [{"id": MODEL, "object": "model"}]})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            return self._send(404, {"error": "not found"})
        if not self._authorized():
            return self._send(401, {"error": "unauthorized"})
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt = " ".join(m.get("content", "") for m in req.get("messages", []))
        prompt_tokens = max(1, len(prompt) // 4)          # rough 4 chars/token
        max_tokens = int(req.get("max_tokens", 64))
        answer = ("This is a mock answer. Replace the mock server with vLLM, "
                  "llama.cpp or Ollama for real output. ") * 8
        completion_tokens = min(max_tokens, len(answer) // 4)
        with slots:                                     # imitate batch capacity
            time.sleep(prompt_tokens / 1000 * PREFILL_S_PER_1K)
            time.sleep(completion_tokens * DECODE_S_PER_TOKEN)
        with lock:
            stats["requests"] += 1
            stats["tokens_out"] += completion_tokens
        self._send(200, {
            "id": "chatcmpl-mock", "object": "chat.completion", "model": MODEL,
            "choices": [{"index": 0, "finish_reason": "length",
                         "message": {
                             "role": "assistant",
                             "content": answer[:completion_tokens * 4]}}],
            "usage": {"prompt_tokens": prompt_tokens,
                      "completion_tokens": completion_tokens,
                      "total_tokens": prompt_tokens + completion_tokens}})

if __name__ == "__main__":
    port = int(os.environ.get("MOCK_PORT", "8000"))
    print(f"mock LLM on http://127.0.0.1:{port}  (key={API_KEY})")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
