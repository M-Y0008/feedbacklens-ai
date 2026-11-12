"""超时修复验证（不写数据库）。

1) 本地慢服务器 10 秒不响应，socket 超时 30 秒、墙钟上限 1 秒：连续两条都应在约 1 秒中断，
   且第二条能继续执行 —— 证明硬超时生效，不依赖 socket 超时。
2) 真实 Ollama 用默认 180 秒上限跑一条正常评论，确认能正常返回。
"""
from __future__ import annotations

import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.analysis import build_system_prompt, build_user_prompt  # noqa: E402
from core.db import init_db  # noqa: E402
from core.llm import LLMCallError, OllamaBackend, load_config  # noqa: E402


class SlowHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        time.sleep(10)
        body = json.dumps({"message": {"content": '{"topics":["other"]}'}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def main() -> int:
    conn = init_db(ROOT / "data" / "feedbacklens.db")
    row = conn.execute(
        "SELECT content, score, p.name AS app FROM feedback f "
        "JOIN products p ON p.id=f.product_id WHERE f.id=1"
    ).fetchone()
    content, score, app = row["content"], row["score"], row["app"]
    cfg = load_config()
    system = build_system_prompt(cfg)
    user = build_user_prompt(app, score, content)

    server = ThreadingHTTPServer(("127.0.0.1", 0), SlowHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    slow = OllamaBackend({
        "base_url": f"http://127.0.0.1:{server.server_address[1]}",
        "chat_path": "/api/chat",
        "model": "qwen3:8b",
        "timeout_seconds": 30,   # 故意让 socket 超时失效
        "max_wall_seconds": 1,   # 只靠墙钟硬上限
    })
    slow_calls = []
    for i in range(2):
        t0 = time.perf_counter()
        try:
            slow.chat_json(system, user)
            code = "no_timeout"
        except LLMCallError as e:
            code = e.code
        slow_calls.append({"call": i + 1, "code": code,
                           "seconds": round(time.perf_counter() - t0, 2)})
    server.shutdown()

    real = OllamaBackend(cfg["backends"]["ollama"])
    t0 = time.perf_counter()
    ok, code = True, ""
    try:
        resp = real.chat_json(system, user)
    except LLMCallError as e:
        ok, code = False, e.code
    real_call = {"ok": ok, "code": code, "seconds": round(time.perf_counter() - t0, 2),
                 "default_timeout": real.timeout, "default_max_wall": real.max_wall}
    if ok:
        real_call["output_chars"] = len(resp.text)

    print(json.dumps({"slow_calls": slow_calls, "real_call": real_call},
                     ensure_ascii=False, indent=2))
    conn.close()
    return 0 if all(c["code"] == "wall_timeout" for c in slow_calls) and ok else 1


if __name__ == "__main__":
    sys.exit(main())
