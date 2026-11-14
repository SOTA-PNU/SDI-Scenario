"""Logging reverse proxy between an agent (furio/openclaude) and the NPU router.

Every request and response passes through unchanged (streamed responses are forwarded as
they arrive) and is written to --log-dir:
  <NNNN>_request.json   method, path, headers (auth redacted), exact body
  <NNNN>_response.raw   the exact response bytes (SSE text for streamed calls)
  index.jsonl           one line per exchange: times, status, sizes, injected flag
--inject-logprobs adds `logprobs: true, top_logprobs: 1` to chat completions that do not ask
for them, so the generated text itself is kept even when the server drops a tool call it
cannot parse (REPORT.md 실험 11); sampling is unaffected.

    python3 llm_log_proxy.py --port 8410 --upstream 127.0.0.1:8400 --log-dir <dir> [--inject-logprobs]
"""
import argparse
import http.client
import itertools
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

A = None
SEQ = itertools.count(1)
LOCK = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        self.forward("GET")

    def do_POST(self):
        self.forward("POST")

    def forward(self, method):
        with LOCK:
            n = next(SEQ)
        t0 = time.time()
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        injected = False
        if A.inject_logprobs and method == "POST" and self.path.endswith("/chat/completions"):
            try:
                j = json.loads(body)
                if "logprobs" not in j:
                    j["logprobs"] = True
                    j["top_logprobs"] = 1
                    body = json.dumps(j).encode()
                    injected = True
            except ValueError:
                pass
        hdr_log = {k: ("<redacted>" if k.lower() in ("authorization", "x-api-key") else v)
                   for k, v in self.headers.items()}
        try:
            req_body = json.loads(body) if body else None
        except ValueError:
            req_body = body.decode(errors="replace")
        base = os.path.join(A.log_dir, f"{n:04d}")
        with open(base + "_request.json", "w") as f:
            json.dump({"seq": n, "time": time.strftime("%F %T"), "t_epoch": t0, "method": method,
                       "path": self.path, "headers": hdr_log, "injected_logprobs": injected,
                       "body": req_body}, f, ensure_ascii=False, indent=1)
        host, port = A.upstream.split(":")
        conn = http.client.HTTPConnection(host, int(port), timeout=A.timeout)
        hdrs = {k: v for k, v in self.headers.items()
                if k.lower() not in ("host", "content-length", "connection", "accept-encoding")}
        hdrs["Content-Length"] = str(len(body))
        hdrs["Connection"] = "close"
        status, nbytes, err = None, 0, None
        try:
            conn.request(method, self.path, body=body if body else None, headers=hdrs)
            resp = conn.getresponse()
            status = resp.status
            self.send_response(resp.status)
            for k, v in resp.getheaders():
                if k.lower() in ("transfer-encoding", "connection", "content-length"):
                    continue
                self.send_header(k, v)
            self.send_header("Connection", "close")
            self.end_headers()
            client_ok = True
            with open(base + "_response.raw", "wb") as f:
                while True:
                    chunk = resp.read1(65536)
                    if not chunk:
                        break
                    f.write(chunk)
                    f.flush()
                    nbytes += len(chunk)
                    if client_ok:
                        try:
                            self.wfile.write(chunk)
                            self.wfile.flush()
                        except (BrokenPipeError, ConnectionResetError):
                            client_ok = False
        except Exception as e:  # upstream down or timeout: tell the client, keep the record
            err = f"{type(e).__name__}: {e}"
            if status is None:
                try:
                    self.send_response(502)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": err}).encode())
                except Exception:
                    pass
        finally:
            conn.close()
            self.close_connection = True
        with LOCK:
            with open(os.path.join(A.log_dir, "index.jsonl"), "a") as f:
                f.write(json.dumps({"seq": n, "time": time.strftime("%F %T"), "method": method, "path": self.path,
                                    "status": status, "req_bytes": len(body), "resp_bytes": nbytes,
                                    "wall_s": round(time.time() - t0, 3), "injected_logprobs": injected,
                                    "error": err}, ensure_ascii=False) + "\n")


def main():
    global A
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8410)
    ap.add_argument("--upstream", default="127.0.0.1:8400")
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--inject-logprobs", action="store_true")
    ap.add_argument("--timeout", type=float, default=1800)
    A = ap.parse_args()
    os.makedirs(A.log_dir, exist_ok=True)
    ThreadingHTTPServer(("127.0.0.1", A.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
