"""Tiny MJPEG-over-HTTP server for the carter live view.

Serves the newest complete JPEG from /dev/shm/carter_live as a
multipart/x-mixed-replace stream.  Pure stdlib, no isaac deps, CPU-only.

Usage: python3 mjpeg_server.py [port]   (default 49100 - already open in ufw)
"""
import glob
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

FRAME_DIR = "/dev/shm/carter_live"
BOUNDARY = "carterframe"

PAGE = b"""<!doctype html><title>Carter Live</title>
<style>body{margin:0;background:#111;display:flex;justify-content:center}
img{max-width:100vw;max-height:100vh}</style>
<img src="/stream">"""


def newest_jpeg():
    """Return bytes of the newest complete JPEG (SOI..EOI), or None."""
    files = sorted(
        glob.glob(os.path.join(FRAME_DIR, "*.jpg")),
        key=os.path.getmtime,
        reverse=True,
    )
    for path in files[:3]:
        try:
            with open(path, "rb") as f:
                data = f.read()
        except OSError:
            continue
        if len(data) > 1000 and data[:2] == b"\xff\xd8" and data[-2:] == b"\xff\xd9":
            return data
    return None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/stream":
            self.send_response(200)
            self.send_header(
                "Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}"
            )
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            last = None
            try:
                while True:
                    data = newest_jpeg()
                    if data and data != last:
                        last = data
                        self.wfile.write(
                            b"--" + BOUNDARY.encode() + b"\r\n"
                            b"Content-Type: image/jpeg\r\n"
                            b"Content-Length: " + str(len(data)).encode() + b"\r\n\r\n"
                        )
                        self.wfile.write(data)
                        self.wfile.write(b"\r\n")
                    time.sleep(0.05)
            except (BrokenPipeError, ConnectionResetError):
                return
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(PAGE)))
            self.end_headers()
            self.wfile.write(PAGE)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 49100
    print(f"[mjpeg] serving http://0.0.0.0:{port}  (frames: {FRAME_DIR})", flush=True)
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
