"""本地网页：训练时打开它，损失曲线每秒刷新。"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def start_monitor(run_dir: Path, host: str, port: int) -> tuple[ThreadingHTTPServer, int]:
    html = (Path(__file__).resolve().parent / "monitor.html").read_bytes()
    summary_path = run_dir / "summary.json"
    metrics_path = run_dir / "metrics.jsonl"

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                self._send(200, "text/html; charset=utf-8", html)
            elif path == "/api/summary":
                body = summary_path.read_bytes() if summary_path.is_file() else b"{}"
                self._send(200, "application/json; charset=utf-8", body)
            elif path == "/api/metrics":
                body = json.dumps(read_jsonl(metrics_path)).encode("utf-8")
                self._send(200, "application/json; charset=utf-8", body)
            else:
                self._send(404, "text/plain; charset=utf-8", b"not found")

        def _send(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args) -> None:
            return

    ThreadingHTTPServer.allow_reuse_address = True
    last_error = None
    for candidate in range(port, port + 20):
        try:
            server = ThreadingHTTPServer((host, candidate), Handler)
            return server, candidate
        except OSError as error:
            last_error = error
    raise RuntimeError(f"从 {port} 起没有可用端口：{last_error}")
