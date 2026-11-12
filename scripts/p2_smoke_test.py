"""P2 前置：qwen3:8b 最小结构化输出验证（3–5 条真实评论）。

产出：reports/qwen3_8b_smoke_test.md，并打印 JSON 摘要。
不写 analyses 表（P2 才做全量落库），本脚本只验证模型行为与耗时。
"""
from __future__ import annotations

import json
import os
import sqlite3
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "feedbacklens.db"
OLLAMA_URL = "http://localhost:11434/api/chat"
# 目标模型 qwen3:8b；可用环境变量 SMOKE_MODEL 覆盖（仅用于链路验证）
MODEL = os.environ.get("SMOKE_MODEL", "qwen3:8b")
OUT_MD = ROOT / "reports" / f"smoke_test_{MODEL.replace(':', '_')}.md"

SCHEMA_HINT = """Return ONLY a JSON object, no markdown, no explanation:
{
  "themes": ["one or more from: account_login, pricing_subscription, answer_quality, speed_latency, stability_crash, missing_features, ui_usability, privacy_security, customer_support, sync_platform, model_limits, other"],
  "sentiment": "positive|neutral|mixed|negative",
  "intent": "praise|complaint|feature_request|question|other",
  "pain_point": "short phrase or null",
  "feature_request": "short phrase or null",
  "severity": 1,
  "evidence_quote": "an exact contiguous substring copied from the review",
  "confidence": 0.0,
  "needs_human_review": false
}"""

PROMPT = """You are a product analytics assistant. Analyze ONE mobile app review.

Rules:
- severity is an integer 1-5 (1=minor annoyance, 5=blocks usage or user churn).
- evidence_quote MUST be copied verbatim from the review text (contiguous substring, same words, same order). Never paraphrase.
- If the review is too short or ambiguous, set confidence < 0.5 and needs_human_review = true.
- Keep pain_point / feature_request short (<= 12 words) or null.

""" + SCHEMA_HINT + "\n\nReview (app: {app}, star rating: {score}):\n\"\"\"{content}\"\"\""


def pick_reviews(n: int = 4) -> list[dict]:
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT f.id, f.content, f.score, p.name AS app FROM feedback f JOIN products p ON p.id=f.product_id "
        "ORDER BY f.score, f.id"
    ).fetchall()
    conn.close()
    by_score: dict[int, list[dict]] = {}
    for r in rows:
        by_score.setdefault(int(r["score"]), []).append(dict(r))
    want = [1, 3, 4, 5][:n]
    picked = []
    for s in want:
        bucket = by_score.get(s) or []
        if bucket:
            picked.append(bucket[len(bucket) // 2])
    return picked


def gpu_mem_mb() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip().splitlines()
        return int(float(out[0])) if out else None
    except Exception:
        return None


def build_prompt(app: str, score: int, content: str) -> str:
    return (PROMPT
            .replace("{app}", str(app))
            .replace("{score}", str(score))
            .replace("{content}", content))


def call_model(app: str, score: int, content: str) -> tuple[str, float]:
    payload = {
        "model": MODEL,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0, "num_ctx": 4096},
        "messages": [{"role": "user", "content": build_prompt(app, score, content)}],
    }
    req = urllib.request.Request(
        OLLAMA_URL, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=180) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    elapsed = time.perf_counter() - t0
    return body.get("message", {}).get("content", ""), elapsed


def validate(raw: str, content: str) -> tuple[dict, list[str]]:
    errors = []
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        return {}, [f"JSON 解析失败：{e}"]
    for k in ("themes", "sentiment", "intent", "severity", "evidence_quote", "confidence", "needs_human_review"):
        if k not in obj:
            errors.append(f"缺字段 {k}")
    quote = str(obj.get("evidence_quote") or "")
    if quote and quote not in content:
        errors.append("evidence_quote 不是原文连续子串")
    if obj.get("sentiment") not in {"positive", "neutral", "mixed", "negative"}:
        errors.append("sentiment 取值非法")
    if obj.get("intent") not in {"praise", "complaint", "feature_request", "question", "other"}:
        errors.append("intent 取值非法")
    if not isinstance(obj.get("severity"), int) or not 1 <= obj.get("severity", 0) <= 5:
        errors.append("severity 非 1-5 整数")
    return obj, errors


def main() -> None:
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5)
    except urllib.error.URLError as e:
        raise SystemExit(f"Ollama 未运行：{e}")

    reviews = pick_reviews(4)
    results = []
    for r in reviews:
        try:
            raw, sec = call_model(r["app"], r["score"], r["content"])
        except Exception as e:  # 单条失败隔离
            results.append({**r, "error": str(e)})
            continue
        obj, errs = validate(raw, r["content"])
        results.append({
            "app": r["app"], "score": r["score"], "content": r["content"],
            "raw": raw, "parsed": obj, "errors": errs, "seconds": round(sec, 2),
            "vram_mb": gpu_mem_mb(),
        })

    ok = [x for x in results if not x.get("errors") and "error" not in x]
    times = [x["seconds"] for x in results if "seconds" in x]
    summary = {
        "model": MODEL,
        "n": len(results),
        "structured_ok": len(ok),
        "success_rate": f"{len(ok)/len(results)*100:.0f}%" if results else "n/a",
        "quote_hit_rate": f"{sum(1 for x in ok if x['parsed'].get('evidence_quote') in x['content'])/max(1,len(ok))*100:.0f}%",
        "latency_s": {"min": min(times), "median": round(statistics.median(times), 2), "max": max(times)} if times else None,
        "vram_mb_peak": max([x["vram_mb"] for x in results if x.get("vram_mb")] or [0]) or None,
        "results": results,
    }
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# qwen3:8b 最小结构化输出验证（实测）\n",
        f"- 模型：`{MODEL}`；样本：{len(results)} 条真实评论（来自 P1 抽样库）",
        f"- 结构化成功：{summary['structured_ok']}/{summary['n']}",
        f"- 引文命中率（严格子串）：{summary['quote_hit_rate']}",
        f"- 单条耗时：{summary['latency_s']}",
        f"- 显存占用峰值：{summary['vram_mb_peak']} MB",
        "",
        "## 逐条结果\n",
    ]
    for i, x in enumerate(results, 1):
        lines.append(f"### {i}. {x.get('app')} / {x.get('score')}★")
        lines.append(f"- 原文：{x.get('content')}")
        if "error" in x:
            lines.append(f"- 调用失败：{x['error']}")
        else:
            lines.append(f"- 耗时：{x['seconds']} s；显存：{x['vram_mb']} MB")
            lines.append(f"- AI 输出：`{json.dumps(x['parsed'], ensure_ascii=False)}`")
            lines.append(f"- 校验问题：{'无' if not x['errors'] else '；'.join(x['errors'])}")
        lines.append("")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.exit(main())
