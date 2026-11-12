"""在线免费模型 vs 本地 qwen3:8b：60 条人工标注的同口径对比。

用法（无 key 先演练，不联网、不跑本地推理）：
    python scripts/p2_online_compare.py --mock

key 到位后（base_url / model / key 见 docs/INSTALL_AND_USAGE.md「在线免费模型对比」）：
    python scripts/p2_online_compare.py

隔离保证：
- analyses 幂等键为 (feedback_id, model_name, prompt_version)，新模型写自己的 model_name，
  不会覆盖或删除 qwen3:8b 的 v2.2-p2（450 条）与 v2.3（60 条）。
- 脚本会对比运行前后的既有版本行数，不一致直接报错。
- --mock 默认使用 data/feedbacklens.mock.db（首次从真实库复制），不写真实库。
- 真实模式写库前自动 .bak 备份。
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shutil
import sqlite3
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.analysis import normalize_ws  # noqa: E402
from core.batch import run_batch  # noqa: E402
from core.db import backup_if_exists, init_db  # noqa: E402
from core.llm import LLMConfigError, get_backend, load_config  # noqa: E402

METRIC_KEYS = [
    ("theme_hit_rate", "主题命中率（至少一个）"),
    ("theme_exact_rate", "主题完全一致"),
    ("sentiment_acc", "情感准确率"),
    ("intent_acc", "意图准确率"),
    ("severity_within1", "严重度 ±1"),
    ("all_fields_exact", "四字段全对"),
    ("quote_hit_rate", "引文命中率"),
]


# ---------------------------------------------------------------- 数据读取
def load_labels(conn: sqlite3.Connection, project_id: int, labeler: str,
                sample_size: int | None) -> list[dict]:
    sql = (
        "SELECT l.feedback_id, l.gold_themes_json, l.gold_sentiment, l.gold_intent, "
        "l.gold_severity, l.sample_size, f.review_id, f.content, f.score, "
        "p.name AS app "
        "FROM eval_labels l JOIN feedback f ON f.id=l.feedback_id "
        "JOIN products p ON p.id=f.product_id "
        "WHERE f.project_id=? AND l.labeler=?"
    )
    params: list = [project_id, labeler]
    if sample_size:
        sql += " AND l.sample_size=?"
        params.append(sample_size)
    sql += " ORDER BY l.feedback_id"
    return [dict(r) for r in conn.execute(sql, params)]


def analysis_rows(conn: sqlite3.Connection, ids: list[int], model: str,
                  version: str) -> dict[int, dict]:
    if not ids:
        return {}
    marks = ",".join("?" * len(ids))
    rows = conn.execute(
        "SELECT feedback_id, themes_json, sentiment, intent, severity, "
        "evidence_quote, status FROM analyses "
        f"WHERE model_name=? AND prompt_version=? AND feedback_id IN ({marks})",
        [model, version, *ids],
    )
    return {int(r["feedback_id"]): dict(r) for r in rows}


def snapshot_counts(conn: sqlite3.Connection, model: str, version: str) -> tuple[int, int]:
    n = conn.execute(
        "SELECT COUNT(*) FROM analyses WHERE model_name=? AND prompt_version=?",
        (model, version),
    ).fetchone()[0]
    total = conn.execute("SELECT COUNT(*) FROM analyses").fetchone()[0]
    return int(n), int(total)


def _parse_themes(raw) -> list[str]:
    try:
        t = json.loads(raw or "[]")
        return [str(x) for x in t] if isinstance(t, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


def _sev_ok(gold, ai) -> bool:
    if gold is None or ai is None:
        return False
    try:
        return abs(float(gold) - float(ai)) <= 1
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------- 指标
def compute_metrics(labels: list[dict], ai_map: dict[int, dict]) -> dict:
    """与 app/ui/data.compute_label_metrics 完全同口径（60 条为分母，缺失按不命中）。"""
    n = len(labels)
    hit = exact = sent = intent = sev = allok = quote = covered = 0
    confusion: dict[str, dict[str, int]] = {}
    for lab in labels:
        ai = ai_map.get(int(lab["feedback_id"]))
        if ai is None:
            continue
        covered += 1
        gold = _parse_themes(lab["gold_themes_json"])
        got = _parse_themes(ai["themes_json"])
        h = bool(set(gold) & set(got))
        e = set(gold) == set(got)
        s = lab["gold_sentiment"] == ai["sentiment"]
        i = lab["gold_intent"] == ai["intent"]
        v = _sev_ok(lab["gold_severity"], ai["severity"])
        q = bool(ai["evidence_quote"]) and \
            normalize_ws(ai["evidence_quote"]) in normalize_ws(lab["content"])
        hit += h
        exact += e
        sent += s
        intent += i
        sev += v
        allok += (e and s and i and v)
        quote += q
        g = str(lab["gold_sentiment"])
        a = str(ai["sentiment"])
        confusion.setdefault(g, {}).setdefault(a, 0)
        confusion[g][a] += 1
    denom = n or 1
    return {
        "labeled": n,
        "covered": covered,
        "theme_hit_rate": round(hit / denom * 100, 1),
        "theme_exact_rate": round(exact / denom * 100, 1),
        "sentiment_acc": round(sent / denom * 100, 1),
        "intent_acc": round(intent / denom * 100, 1),
        "severity_within1": round(sev / denom * 100, 1),
        "all_fields_exact": round(allok / denom * 100, 1),
        "quote_hit_rate": round(quote / denom * 100, 1),
        "_confusion": confusion,
    }


def diff_summary(labels: list[dict], base_map: dict[int, dict],
                 new_map: dict[int, dict]) -> dict:
    """逐条对比新旧模型在四个字段上的变化条数（只统计两边都有结果的条目）。"""
    out = {"both": 0, "themes_changed": 0, "sentiment_changed": 0,
           "intent_changed": 0, "severity_changed": 0}
    for lab in labels:
        fid = int(lab["feedback_id"])
        a, b = base_map.get(fid), new_map.get(fid)
        if not a or not b:
            continue
        out["both"] += 1
        out["themes_changed"] += _parse_themes(a["themes_json"]) != _parse_themes(b["themes_json"])
        out["sentiment_changed"] += a["sentiment"] != b["sentiment"]
        out["intent_changed"] += a["intent"] != b["intent"]
        out["severity_changed"] += a["severity"] != b["severity"]
    return out


# ---------------------------------------------------------------- 统计与明细
def _correct_flags(lab: dict, ai: dict) -> dict:
    gold = _parse_themes(lab["gold_themes_json"])
    got = _parse_themes(ai["themes_json"])
    e = set(gold) == set(got)
    s = lab["gold_sentiment"] == ai["sentiment"]
    i = lab["gold_intent"] == ai["intent"]
    v = _sev_ok(lab["gold_severity"], ai["severity"])
    return {"themes": e, "sentiment": s, "intent": i, "severity": v,
            "all_fields": e and s and i and v}


def _value_text(lab: dict, ai: dict, field: str) -> tuple[str, str]:
    if field == "themes":
        return ", ".join(_parse_themes(lab["gold_themes_json"])) or "-", \
               ", ".join(_parse_themes(ai["themes_json"])) or "-"
    if field == "all_fields":
        return _gold_summary_text(lab), _summary_text(ai)
    return str(lab.get(f"gold_{field}")), str(ai.get(field))


def _gold_summary_text(lab: dict) -> str:
    return (f'{lab.get("gold_sentiment")}/{lab.get("gold_intent")}/'
            f'sev={lab.get("gold_severity")}/'
            f'{",".join(_parse_themes(lab.get("gold_themes_json") or "[]")) or "-"}')


def _summary_text(row: dict) -> str:
    return (f'{row.get("sentiment")}/{row.get("intent")}/'
            f'sev={row.get("severity")}/'
            f'{",".join(_parse_themes(row.get("themes_json") or "[]")) or "-"}')


def flip_details(labels: list[dict], base_map: dict[int, dict],
                 new_map: dict[int, dict]) -> dict:
    """改对/改错清单：相对人工标注，新模型修好/弄坏了哪些字段。"""
    out = {f: {"fixed": [], "broken": []} for f in
           ("all_fields", "themes", "sentiment", "intent", "severity")}
    for lab in labels:
        fid = int(lab["feedback_id"])
        a, b = base_map.get(fid), new_map.get(fid)
        if not a or not b:
            continue
        ca, cb = _correct_flags(lab, a), _correct_flags(lab, b)
        for field, ok in ca.items():
            if ok == cb[field]:
                continue
            gold, _ = _value_text(lab, a, field)
            _, old = _value_text(lab, a, field)
            _, new = _value_text(lab, b, field)
            item = {
                "review_id": lab["review_id"],
                "app": lab["app"],
                "excerpt": (lab["content"] or "").replace("\n", " ")[:70],
                "gold": gold, "base": old, "new": new,
            }
            out[field]["fixed" if cb[field] else "broken"].append(item)
    return out


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return round(max(0.0, centre - margin) * 100, 1), round(min(1.0, centre + margin) * 100, 1)


def mcnemar_exact(b: int, c: int) -> float:
    """配对二分类的 McNemar 精确检验（双侧，二项分布）。b/c 为不一致对数。"""
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return round(min(1.0, p), 4)


def latency_stats(seconds_list: list[float]) -> dict:
    vals = sorted(x for x in seconds_list if x and x > 0)
    if not vals:
        return {"n": 0, "p50": 0.0, "p90": 0.0, "max": 0.0, "mean": 0.0}
    def pct(p: float) -> float:
        return vals[min(len(vals) - 1, max(0, int(round((len(vals) - 1) * p))))]
    return {"n": len(vals), "p50": round(pct(0.5), 2), "p90": round(pct(0.9), 2),
            "max": round(vals[-1], 2), "mean": round(sum(vals) / len(vals), 2)}


def mask_key(key: str) -> str:
    return (key[:4] + "****") if key and len(key) >= 4 else "****"


# ---------------------------------------------------------------- mock 服务
def _mock_output(content: str, star: int | None) -> dict:
    """本地 stub 的确定性输出，仅用于验证接口与解析，不代表任何真实模型能力。"""
    text = content or ""
    low = text.lower()
    topics: list[str] = []
    rules = [
        ("stability_crash", ("crash", "freeze", "froze", "force close", "won't open")),
        ("speed_latency", ("slow", "lag", "wait", "delay", "timeout")),
        ("pricing_subscription", ("price", "expensive", "subscription", "paywall", "charge")),
        ("answer_quality", ("wrong", "inaccurate", "hallucinat", "irrelevant", "off-topic")),
        ("account_login", ("login", "sign in", "sign-in", "account", "verify")),
        ("missing_features", ("please add", "wish", "missing", "not available", "bring back")),
        ("ui_usability", ("interface", "layout", "hard to find", "confusing", "design")),
        ("privacy_security", ("privacy", "data", "permission")),
        ("customer_support", ("support", "refund", "ticket")),
        ("sync_platform", ("sync", "device", "web version")),
        ("model_limits", ("context", "cannot browse", "can't browse", "language limit")),
    ]
    for code, words in rules:
        if any(w in low for w in words):
            topics.append(code)
        if len(topics) >= 2:
            break
    if not topics:
        topics = ["other"]

    prompt = text.strip()
    quote = prompt[:60].strip() if prompt else ""
    if len(quote) < 4:
        quote = prompt

    if star is None:
        sentiment = "neutral"
    elif star <= 2:
        sentiment = "negative"
    elif star >= 4:
        sentiment = "positive"
    else:
        sentiment = "mixed"

    if any(w in low for w in ("please add", "wish", "hope", "bring back")):
        intent = "feature_request"
    elif "?" in text:
        intent = "question"
    elif star is not None and star <= 2:
        intent = "complaint"
    elif star is not None and star >= 4:
        intent = "praise"
    else:
        intent = "other"

    severity = {1: 5, 2: 4, 3: 2, 4: 1, 5: 0}.get(star or 3, 2)
    return {
        "topics": topics,
        "sentiment": sentiment,
        "intent": intent,
        "pain_point": None if not quote else "mock pain point",
        "feature_request": None,
        "severity": severity,
        "evidence_quote": quote,
        "confidence": 0.8,
        "needs_human_review": False,
    }


class _PacedBackend:
    """在 backend 外包一层，强制两次请求之间的最小间隔，避免触发免费额度限速。"""

    def __init__(self, inner, min_interval: float):
        self.inner = inner
        self.min_interval = max(0.0, float(min_interval))
        self._last = 0.0

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def chat_json(self, system: str, user: str):
        gap = self.min_interval - (time.perf_counter() - self._last)
        if gap > 0:
            time.sleep(gap)
        try:
            return self.inner.chat_json(system, user)
        finally:
            self._last = time.perf_counter()


class _StubHandler(BaseHTTPRequestHandler):
    state: dict = {}

    def log_message(self, *args) -> None:  # 静默
        pass

    def _send(self, code: int, obj: dict) -> None:
        raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return self._send(400, {"error": {"message": "bad json"}})
        auth = self.headers.get("Authorization") or ""
        if not auth.startswith("Bearer ") or len(auth) <= len("Bearer "):
            return self._send(401, {"error": {"message": "missing api key"}})
        with self.state["lock"]:
            self.state["calls"] += 1
            call_no = self.state["calls"]
        # 第一条先返回 429，验证重试分支确实生效
        if call_no == 1:
            return self._send(429, {"error": {"message": "rate limited (mock)"}})
        messages = body.get("messages") or []
        user = str(messages[-1].get("content") or "") if messages else ""
        m = re.search(r'"""(.*?)"""', user, re.S)
        content = m.group(1) if m else user
        star_m = re.search(r"star rating:\s*(\d+)", user)
        star = int(star_m.group(1)) if star_m else None
        obj = _mock_output(content, star)
        self._send(200, {
            "id": "mock-completion",
            "object": "chat.completion",
            "model": body.get("model") or "mock:stub",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": json.dumps(obj, ensure_ascii=False)},
                "finish_reason": "stop",
            }],
        })


def start_stub() -> ThreadingHTTPServer:
    _StubHandler.state = {"calls": 0, "lock": threading.Lock()}
    server = ThreadingHTTPServer(("127.0.0.1", 0), _StubHandler)
    threading.Thread(target=server.serve_forever, daemon=True, name="mock-api").start()
    return server


# ---------------------------------------------------------------- 报告
def write_report(path: Path, *, mock: bool, db_path: Path, labeler: str,
                 baseline: dict, baseline_metrics: dict, new_metrics: dict,
                 backend, prompt_version: str, run_id: str, stats,
                 diff: dict, guard: dict, extra: dict | None = None) -> None:
    extra = extra or {}
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    lines = [
        f"# 在线模型 vs 本地模型 · {baseline_metrics['labeled']} 条标注对比"
        f"{'（MOCK 演练）' if mock else ''}",
        "",
        f"- 生成时间（UTC）：{now}",
        f"- 数据库：`{db_path}`",
        f"- 标注人：{labeler}；样本量：{baseline_metrics['labeled']} 条",
        f"- 基线：`{baseline['model']}` / `{baseline['version']}`",
        f"- 新模型：`{backend.model}` / `{prompt_version}`（后端 `{backend.name}`）",
        f"- run_id：`{run_id}`",
    ]
    if mock:
        lines += [
            "",
            "> **MOCK 演练**：本轮使用本地 stub 服务，没有调用任何真实在线模型。"
            "表中数字只证明链路、解析、指标与落库可运行，**不能作为质量结论**。",
        ]
    lines += [
        "",
        "## 1. 指标对比",
        "",
        "| 指标 | 基线 | 新模型 | 变化 |",
        "|---|---|---|---|",
    ]
    for key, label in METRIC_KEYS:
        a = baseline_metrics.get(key)
        b = new_metrics.get(key)
        delta = round(b - a, 1) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else "-"
        delta = f"{delta:+.1f}pp" if isinstance(delta, float) else delta
        lines.append(f"| {label} | {a}% | {b}% | {delta} |")
    lines += [
        f"| 有结果条数 | {baseline_metrics['covered']} | {new_metrics['covered']} | "
        f"{new_metrics['covered'] - baseline_metrics['covered']:+d} |",
        "",
        "## 2. 本次运行状态",
        "",
        f"- done {stats.get('done', 0)} / degraded {stats.get('degraded', 0)} / "
        f"failed {stats.get('failed', 0)} / 规则跳过 {stats.get('skipped', 0)} / "
        f"续跑跳过 {stats.get('resume_skipped', 0)}",
        f"- HTTP 请求 {extra.get('usage', {}).get('http_requests', 0)} 次；"
        f"429 {extra.get('usage', {}).get('http_429', 0)} 次；"
        f"重试 {extra.get('usage', {}).get('retries', 0)} 次",
        f"- tokens：输入 {extra.get('usage', {}).get('prompt_tokens', 0)} / "
        f"输出 {extra.get('usage', {}).get('completion_tokens', 0)} / "
        f"合计 {extra.get('usage', {}).get('total_tokens', 0)}",
        f"- 总耗时 {extra.get('wall_seconds', 0)} 秒；单条延迟（秒）"
        f" p50 {extra.get('latency', {}).get('p50', 0)} / "
        f"p90 {extra.get('latency', {}).get('p90', 0)} / "
        f"max {extra.get('latency', {}).get('max', 0)}",
        f"- 限速节奏：每条请求间隔 ≥ {extra.get('pace_seconds', 0)} 秒",
        "",
        "## 3. 与基线的逐条差异",
        "",
        f"- 两边都有结果：{diff['both']} 条",
        f"- 主题集合变化：{diff['themes_changed']} 条；情感变化：{diff['sentiment_changed']} 条；"
        f"意图变化：{diff['intent_changed']} 条；严重度变化：{diff['severity_changed']} 条",
    ]
    flips = extra.get("flips") or {}
    lines += ["", "### 3.1 四字段全对：改对 / 改错", ""]
    fixed = (flips.get("all_fields") or {}).get("fixed", [])
    broken = (flips.get("all_fields") or {}).get("broken", [])
    lines.append(f"- 基线错、新模型对：{len(fixed)} 条；基线对、新模型错：{len(broken)} 条")
    if fixed or broken:
        lines += ["", "| 结果 | review_id | 产品 | 原文摘录 | 人工标注 | 本地 qwen3:8b | 新模型 |",
                  "|---|---|---|---|---|---|---|"]
        for tag, rows in (("改对", fixed), ("改错", broken)):
            for it in rows:
                lines.append(
                    f"| {tag} | {it['review_id']} | {it['app']} | {it['excerpt']} | "
                    f"{it['gold']} | {it['base']} | {it['new']} |")
    lines += ["", "### 3.2 分字段改对 / 改错条数", "",
              "| 字段 | 改对 | 改错 |", "|---|---|---|"]
    for field, name in (("themes", "主题"), ("sentiment", "情感"),
                        ("intent", "意图"), ("severity", "严重度")):
        f = flips.get(field) or {"fixed": [], "broken": []}
        lines.append(f"| {name} | {len(f['fixed'])} | {len(f['broken'])} |")
    sig = extra.get("significance") or {}
    lines += ["", "### 3.3 统计显著性（四字段全对）", "",
              f"- 配对不一致：基线错→新对 {sig.get('b', 0)} 条，基线对→新错 {sig.get('c', 0)} 条",
              f"- McNemar 精确检验（双侧）p = {sig.get('p', '-')}；"
              f"α=0.05 下{'显著' if isinstance(sig.get('p'), float) and sig['p'] < 0.05 else '不显著'}",
              f"- 基线 {sig.get('base_rate', '-')}%（95% Wilson CI {sig.get('base_ci', '-')}）；"
              f"新模型 {sig.get('new_rate', '-')}%（95% CI {sig.get('new_ci', '-')}）",
              "- 60 条时置信区间很宽，即使 p<0.05 也只能说明这一批的方向，不能外推到全量。",
              "", "### 3.4 归因提示（需人工复核）", "",
              "- 脚本只给事实差异，不自动判定“模型能力提升”还是“提示词口径问题”；"
              "归因请结合原文证据逐条核对：若新模型引文能在原文命中且字段与原文一致，"
              "更可能是能力提升；若引文正确但字段按规则被补齐/降级，则更可能是口径问题。",
              "", "## 4. 数据隔离校验", "",
        "| 检查项 | 运行前 | 运行后 |",
        "|---|---|---|",
        f"| `{baseline['model']}` / `{baseline['version']}` | {guard['before_model']} | "
        f"{guard['after_model']} |",
        f"| `{baseline['model']}` / `v2.3` | {guard['before_v23']} | "
        f"{guard['after_v23']} |",
        f"| analyses 总行数 | {guard['before_total']} | {guard['after_total']} |",
        "",
        "## 5. 局限",
        "",
        "- 60 条为单人标注、无标注者间一致性，指标波动大，只能判断方向，不能外推全量。",
        "- 提示词与字段校验未变，分数差异只来自模型；但不同模型的 JSON 遵从度不同，"
        "degraded 会直接影响引文与四字段指标。",
        "- 免费额度与速率限制会影响可跑条数，若中途 429 失败，重跑会断点续跑。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------- 主流程
def _slug(model: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", model).strip("_") or "model"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--db", default=None, help="默认取 FEEDBACKLENS_DB 或 data/feedbacklens.db")
    ap.add_argument("--project-id", type=int, default=None)
    ap.add_argument("--labeler", default="PM")
    ap.add_argument("--sample-size", type=int, default=20,
                    help="eval_labels.sample_size（20 = 每产品 20 条，共 60）；0 表示不过滤")
    ap.add_argument("--baseline-version", default="v2.2-p2")
    ap.add_argument("--baseline-model", default=None, help="默认取 config 里的 ollama 模型")
    ap.add_argument("--new-version", default=None, help="默认与基线同版本（只换模型）")
    ap.add_argument("--report", default=None)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条标注（0 = 全部）")
    ap.add_argument("--probe", action="store_true",
                    help="连通性探测：只调用 1 条，不写库")
    ap.add_argument("--mock", action="store_true", help="本地 stub 服务演练，不联网")
    ap.add_argument("--mock-db", default=None, help="mock 模式专用库，默认 data/feedbacklens.mock.db")
    ap.add_argument("--mock-model", default="mock:stub")
    ap.add_argument("--min-interval", type=float, default=0.0,
                    help="两次在线请求的最小间隔秒数（按免费额度限速，0 = 不限制）")
    ap.add_argument("--dry-run", action="store_true", help="只校验配置与样本，不调用模型")
    args = ap.parse_args()

    cfg = load_config(args.config)
    real_db = Path(args.db or os.environ.get("FEEDBACKLENS_DB")
                   or ROOT / "data" / "feedbacklens.db")
    db_path = real_db
    if args.mock:
        db_path = Path(args.mock_db or ROOT / "data" / "feedbacklens.mock.db")
        if not db_path.exists():
            if not real_db.exists():
                print(f"[error] 找不到源库：{real_db}", file=sys.stderr)
                return 2
            db_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(real_db, db_path)
            print(f"[mock] 已从 {real_db.name} 复制到 {db_path}")

    # 先校验模型配置：缺 key 时直接给出指引，不打开数据库、不发请求
    baseline_model = args.baseline_model or str(
        cfg.get("backends", {}).get("ollama", {}).get("model", "qwen3:8b"))
    prompt_version = args.new_version or str(cfg.get("prompt_version") or args.baseline_version)
    cfg_new = copy.deepcopy(cfg)
    cfg_new["active_backend"] = "openai_compat"
    cfg_new["prompt_version"] = prompt_version
    backend_cfg = cfg_new.setdefault("backends", {}).setdefault("openai_compat", {})
    backend_cfg["enabled"] = True

    server = None
    if args.mock:
        # 忽略本机已配置的真实 base_url/model，避免 mock 误连线上服务
        for key in ("FEEDBACKLENS_API_BASE", "FEEDBACKLENS_API_MODEL"):
            os.environ.pop(key, None)
        server = start_stub()
        port = server.server_address[1]
        backend_cfg.update({
            "base_url": f"http://127.0.0.1:{port}",
            "chat_path": "/v1/chat/completions",
            "model": args.mock_model,
            "api_key_env": "FEEDBACKLENS_MOCK_KEY",
            "timeout_seconds": 10,
            "max_wall_seconds": 10,
            "max_retries": 1,
            "retry_backoff_seconds": 0.2,
        })
        os.environ["FEEDBACKLENS_MOCK_KEY"] = "mock-key-not-real"

    try:
        backend = get_backend(cfg_new, strict=True)
    except LLMConfigError as e:
        print("[config] " + str(e), file=sys.stderr)
        print("请先设置以下环境变量（API key 只放环境变量，不要写进仓库）：\n"
              '  $env:FEEDBACKLENS_API_BASE="<OpenAI 兼容 base_url>"\n'
              '  $env:FEEDBACKLENS_API_MODEL="<免费模型 id>"\n'
              '  $env:FEEDBACKLENS_API_KEY="<你的 key>"\n'
              "然后重新运行本脚本；不了解平台额度请先看 docs/INSTALL_AND_USAGE.md。",
              file=sys.stderr)
        if server:
            server.shutdown()
        return 2

    if args.min_interval and args.min_interval > 0:
        backend = _PacedBackend(backend, args.min_interval)
        print(f"[pace] 两条请求间隔 ≥ {args.min_interval:.1f} 秒")

    if backend.model == baseline_model and prompt_version == args.baseline_version:
        print("[error] 新模型与基线同名同版本，会覆盖基线结果；请换模型名或换版本。",
              file=sys.stderr)
        if server:
            server.shutdown()
        return 2

    if not db_path.exists():
        print(f"[error] 找不到数据库：{db_path}", file=sys.stderr)
        if server:
            server.shutdown()
        return 2
    try:
        conn = init_db(db_path)
    except sqlite3.OperationalError as e:
        print(f"[error] 无法打开数据库 {db_path}：{e}\n"
              "若提示 disk I/O error，通常是数据库被其他进程占用或存在未恢复的 WAL，"
              "请先关闭占用该库的服务再重试。", file=sys.stderr)
        if server:
            server.shutdown()
        return 2
    conn.row_factory = sqlite3.Row

    project_id = args.project_id
    if project_id is None:
        row = conn.execute("SELECT id FROM projects ORDER BY id LIMIT 1").fetchone()
        if not row:
            print("[error] 数据库里没有项目", file=sys.stderr)
            return 2
        project_id = int(row["id"])

    labels = load_labels(conn, project_id, args.labeler,
                         args.sample_size or None)
    if not labels:
        print(f"[error] 项目 {project_id} 没有标注人 {args.labeler} 的 "
              f"sample_size={args.sample_size} 标注", file=sys.stderr)
        return 2
    if args.limit and args.limit > 0:
        labels = labels[:args.limit]
    print(f"[data] 项目 {project_id} · 标注 {len(labels)} 条 · 库 {db_path}")

    base_map = analysis_rows(conn, [int(x["feedback_id"]) for x in labels],
                             baseline_model, args.baseline_version)
    baseline_metrics = compute_metrics(labels, base_map)
    print(f"[baseline] {baseline_model} / {args.baseline_version} · "
          f"有结果 {baseline_metrics['covered']}/{baseline_metrics['labeled']} · "
          f"四字段 {baseline_metrics['all_fields_exact']}%")

    if args.probe:
        from core.analysis import analyze_one
        item = {"id": int(labels[0]["feedback_id"]),
                "review_id": labels[0]["review_id"],
                "content": labels[0]["content"], "score": labels[0]["score"],
                "app": labels[0]["app"]}
        t0 = time.perf_counter()
        try:
            r = analyze_one(backend, item, cfg_new)
            err = None
        except Exception as e:  # 探测失败立即停止，不重试、不消耗更多额度
            r, err = None, f"{type(e).__name__}: {e}"
        elapsed = round(time.perf_counter() - t0, 2)
        out = {
            "model": backend.model, "base_url": backend.base_url,
            "key": mask_key(getattr(backend, "api_key", "")),
            "elapsed_seconds": elapsed, "usage": backend.stats,
        }
        if err:
            out["error"] = err
            print(json.dumps(out, ensure_ascii=False, indent=2), file=sys.stderr)
            if server:
                server.shutdown()
            return 1
        out.update({"parse_ok": r.get("parse_ok"), "status": r.get("status"),
                    "error": r.get("error"),
                    "valid": r.get("is_valid"), "quote_ok": r.get("quote_ok"),
                    "missing_fields": r.get("missing_fields"),
                    "invalid_fields": r.get("invalid_fields"),
                    "needs_human_review": r.get("needs_human_review"),
                    "parsed_keys": sorted((r.get("raw_parsed") or {}).keys())})
        print(json.dumps(out, ensure_ascii=False, indent=2))
        if server:
            server.shutdown()
        ok = bool(r.get("is_valid")) and bool(r.get("quote_ok"))
        return 0 if ok else 1

    if args.dry_run:
        print(json.dumps({"dry_run": True, "new_model": backend.model,
                          "baseline": baseline_metrics}, ensure_ascii=False, indent=2))
        if server:
            server.shutdown()
        return 0

    before_model, before_total = snapshot_counts(conn, baseline_model, args.baseline_version)
    before_v23 = snapshot_counts(conn, baseline_model, "v2.3")[0]
    bak = backup_if_exists(db_path)
    if bak:
        print(f"[backup] {bak}")

    run_id = ("mock-online-" if args.mock else "online-") + \
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    items = [{"id": int(x["feedback_id"]), "review_id": x["review_id"],
              "content": x["content"], "score": x["score"], "app": x["app"]}
             for x in labels]
    t_run = time.perf_counter()
    stats, details = run_batch(conn, items, backend, cfg_new, run_id)
    wall_seconds = round(time.perf_counter() - t_run, 1)
    print(f"[run] {run_id} · " + json.dumps(dict(stats), ensure_ascii=False)
          + f" · {wall_seconds}s")

    after_model, after_total = snapshot_counts(conn, baseline_model, args.baseline_version)
    after_v23 = snapshot_counts(conn, baseline_model, "v2.3")[0]
    if after_model != before_model or after_v23 != before_v23:
        print(f"[FATAL] 既有版本行数被改变：基线 {before_model}->{after_model}，"
              f"v2.3 {before_v23}->{after_v23}", file=sys.stderr)
        return 3

    new_map = analysis_rows(conn, [int(x["feedback_id"]) for x in labels],
                            backend.model, prompt_version)
    new_metrics = compute_metrics(labels, new_map)
    confusion = new_metrics.pop("_confusion", {})
    diff = diff_summary(labels, base_map, new_map)
    flips = flip_details(labels, base_map, new_map)

    n = len(labels)
    base_ok = new_ok = b = c = 0
    for lab in labels:
        fid = int(lab["feedback_id"])
        a, bb = base_map.get(fid), new_map.get(fid)
        ao = bool(a) and _correct_flags(lab, a)["all_fields"]
        bo = bool(bb) and _correct_flags(lab, bb)["all_fields"]
        base_ok += ao
        new_ok += bo
        b += (not ao) and bo
        c += ao and (not bo)
    sig = {
        "b": b, "c": c, "p": mcnemar_exact(b, c),
        "base_rate": round(base_ok / n * 100, 1) if n else 0,
        "new_rate": round(new_ok / n * 100, 1) if n else 0,
        "base_ci": "%.1f–%.1f" % wilson_ci(base_ok, n),
        "new_ci": "%.1f–%.1f" % wilson_ci(new_ok, n),
    }
    extra = {
        "usage": dict(backend.stats), "wall_seconds": wall_seconds,
        "pace_seconds": args.min_interval,
        "latency": latency_stats([d.get("seconds", 0) for d in details]),
        "flips": flips, "significance": sig,
    }
    created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    conn.execute(
        "INSERT INTO eval_runs (run_id, sample_size, labeler, metrics_json, "
        "confusion_json, model_name, prompt_version, created_at) VALUES (?,?,?,?,?,?,?,?)",
        (run_id, len(labels), args.labeler,
         json.dumps(new_metrics, ensure_ascii=False),
         json.dumps(confusion, ensure_ascii=False),
         backend.model, prompt_version, created_at),
    )
    conn.commit()

    slug = ("mock_" if args.mock else "") + _slug(backend.model)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = Path(args.report) if args.report else (
        ROOT / "reports" / f"online_compare_{slug}_{stamp}.md")
    write_report(
        report_path, mock=args.mock, db_path=db_path, labeler=args.labeler,
        baseline={"model": baseline_model, "version": args.baseline_version},
        baseline_metrics=baseline_metrics, new_metrics=new_metrics,
        backend=backend, prompt_version=prompt_version, run_id=run_id,
        stats=stats, diff=diff,
        guard={"before_model": before_model, "after_model": after_model,
               "before_v23": before_v23, "after_v23": after_v23,
               "before_total": before_total, "after_total": after_total},
        extra=extra,
    )
    if server:
        server.shutdown()
    conn.close()

    print(json.dumps({
        "mock": args.mock,
        "run_id": run_id,
        "db": str(db_path),
        "baseline": {"model": baseline_model, "version": args.baseline_version,
                     **{k: baseline_metrics[k] for k, _ in METRIC_KEYS}},
        "new": {"model": backend.model, "prompt_version": prompt_version,
                **{k: new_metrics[k] for k, _ in METRIC_KEYS}},
        "status": dict(stats),
        "isolation": {"baseline_rows_before": before_model,
                      "baseline_rows_after": after_model},
        "report": str(report_path),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
