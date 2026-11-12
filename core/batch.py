"""批处理与落库（P2）：断点续跑、逐条状态、失败隔离。"""
from __future__ import annotations

import json
import sqlite3
from collections import Counter

from core.analysis import analyze_one
from core.db import utc_now

TERMINAL_STATUS = {"done", "degraded", "skipped"}


def select_trial(conn: sqlite3.Connection, limit: int = 30) -> list[dict]:
    """从已入库样本中确定性地分层抽取：每产品均分名额，按星级均匀取样。"""
    products = conn.execute("SELECT id, name FROM products ORDER BY id").fetchall()
    if not products:
        return []
    per_product = max(1, limit // len(products))
    picked: list[dict] = []
    seen: set[int] = set()
    for p in products:
        rows = conn.execute(
            "SELECT f.id, f.review_id, f.content, f.score, ? AS app "
            "FROM feedback f WHERE f.product_id=? ORDER BY f.score, f.id",
            (p["name"], p["id"]),
        ).fetchall()
        n = len(rows)
        if n == 0:
            continue
        if per_product == 1:
            idx = [0]
        elif n <= per_product:
            idx = list(range(n))
        else:
            idx = sorted({round(i * (n - 1) / (per_product - 1)) for i in range(per_product)})
        for i in idx:
            row = rows[i]
            if row["id"] in seen:
                continue
            seen.add(row["id"])
            picked.append(dict(row))
            if len(picked) >= limit:
                return picked
    return picked


def select_remaining(conn: sqlite3.Connection, model_name: str, old_version: str,
                     new_version: str) -> list[dict]:
    """选出旧版有结果、新版尚无结果的评论，用于把剩余样本补齐到新提示词。"""
    rows = conn.execute(
        "SELECT f.id, f.review_id, f.content, f.score, p.name AS app "
        "FROM analyses a "
        "JOIN feedback f ON f.id = a.feedback_id "
        "JOIN products p ON p.id = f.product_id "
        "WHERE a.model_name=? AND a.prompt_version=? "
        "AND NOT EXISTS ("
        "  SELECT 1 FROM analyses n WHERE n.feedback_id=a.feedback_id "
        "  AND n.model_name=a.model_name AND n.prompt_version=?"
        ") "
        "ORDER BY f.id",
        (model_name, old_version, new_version),
    ).fetchall()
    return [dict(r) for r in rows]


def select_affected(conn: sqlite3.Connection, model_name: str, prompt_version: str) -> list[dict]:
    """挑出旧版提示词下主题仅为 other、或主题含 model_limits 的评论，供定向重跑。"""
    rows = conn.execute(
        "SELECT a.themes_json, f.id, f.review_id, f.content, f.score, p.name AS app "
        "FROM analyses a "
        "JOIN feedback f ON f.id = a.feedback_id "
        "JOIN products p ON p.id = f.product_id "
        "WHERE a.model_name=? AND a.prompt_version=? "
        "ORDER BY f.id",
        (model_name, prompt_version),
    ).fetchall()
    picked: list[dict] = []
    for row in rows:
        try:
            themes = json.loads(row["themes_json"] or "[]")
        except json.JSONDecodeError:
            themes = []
        if not isinstance(themes, list):
            continue
        if themes == ["other"] or "model_limits" in themes:
            picked.append({k: row[k] for k in ("id", "review_id", "content", "score", "app")})
    return picked


def select_theme(conn: sqlite3.Connection, model_name: str, source_version: str,
                 theme: str, target_version: str) -> list[dict]:
    """选出旧版结果中主题含指定主题、且目标版本尚无结果的评论（用于边界调优验证）。"""
    rows = conn.execute(
        "SELECT a.themes_json, f.id, f.review_id, f.content, f.score, p.name AS app "
        "FROM analyses a "
        "JOIN feedback f ON f.id = a.feedback_id "
        "JOIN products p ON p.id = f.product_id "
        "WHERE a.model_name=? AND a.prompt_version=? "
        "AND NOT EXISTS ("
        "  SELECT 1 FROM analyses n WHERE n.feedback_id=a.feedback_id "
        "  AND n.model_name=a.model_name AND n.prompt_version=?"
        ") "
        "ORDER BY f.id",
        (model_name, source_version, target_version),
    ).fetchall()
    picked: list[dict] = []
    for row in rows:
        try:
            themes = json.loads(row["themes_json"] or "[]")
        except json.JSONDecodeError:
            continue
        if isinstance(themes, list) and theme in themes:
            picked.append({k: row[k] for k in ("id", "review_id", "content", "score", "app")})
    return picked


def _save_result(conn: sqlite3.Connection, item: dict, run_id: str, backend, cfg: dict, r: dict) -> None:
    conn.execute(
        """
        INSERT INTO analyses (
            feedback_id, run_id, model_name, model_version, prompt_version,
            themes_json, sentiment, intent, pain_point, feature_request, severity,
            evidence_quote, confidence, needs_human_review, is_valid, raw_output,
            error, status, attempts, created_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(feedback_id, model_name, prompt_version) DO UPDATE SET
            run_id=excluded.run_id,
            themes_json=excluded.themes_json,
            sentiment=excluded.sentiment,
            intent=excluded.intent,
            pain_point=excluded.pain_point,
            feature_request=excluded.feature_request,
            severity=excluded.severity,
            evidence_quote=excluded.evidence_quote,
            confidence=excluded.confidence,
            needs_human_review=excluded.needs_human_review,
            is_valid=excluded.is_valid,
            raw_output=excluded.raw_output,
            error=excluded.error,
            status=excluded.status,
            attempts=analyses.attempts + 1,
            created_at=excluded.created_at
        """,
        (
            item["id"],
            run_id,
            backend.model,
            backend.model,
            cfg["prompt_version"],
            json.dumps(r["themes"], ensure_ascii=False),
            r["sentiment"],
            r["intent"],
            r["pain_point"],
            r["feature_request"],
            r["severity"],
            r["evidence_quote"],
            r["confidence"],
            int(r["needs_human_review"]),
            int(r["is_valid"]),
            (r.get("raw_output") or "")[:4000],
            r["error"],
            r["status"],
            1,
            utc_now(),
        ),
    )


def run_batch(
    conn: sqlite3.Connection,
    items: list[dict],
    backend,
    cfg: dict,
    run_id: str,
    force: bool = False,
) -> tuple[Counter, list[dict]]:
    """逐条分析并落库；已完成/降级/跳过的不再推理，失败的可按 max_attempts 重试。

    force=True 时忽略终态跳过，用于同版本下的定向修复重跑。
    """
    prompt_version = cfg["prompt_version"]
    max_attempts = int(cfg.get("batch", {}).get("max_attempts", 2))
    stats: Counter = Counter()
    details: list[dict] = []

    for item in items:
        prev = conn.execute(
            "SELECT status, attempts FROM analyses "
            "WHERE feedback_id=? AND model_name=? AND prompt_version=?",
            (item["id"], backend.model, prompt_version),
        ).fetchone()
        if prev:
            if prev["status"] in TERMINAL_STATUS and not force:
                stats["resume_skipped"] += 1
                continue
            if int(prev["attempts"] or 0) >= max_attempts and not force:
                stats["max_attempts_skipped"] += 1
                continue

        try:
            r = analyze_one(backend, item, cfg)
        except Exception as e:  # 兜底：单条异常不影响整批
            r = {
                "status": "failed",
                "error": f"unexpected:{type(e).__name__}",
                "is_valid": 0,
                "needs_human_review": 1,
                "themes": [],
                "sentiment": None,
                "intent": None,
                "pain_point": None,
                "feature_request": None,
                "severity": None,
                "evidence_quote": None,
                "confidence": None,
                "flags": [],
                "missing_fields": [],
                "invalid_fields": [],
                "filled_fields": [],
                "quote_ok": False,
                "seconds": 0.0,
                "attempts": 0,
                "raw_output": "",
            }

        _save_result(conn, item, run_id, backend, cfg, r)
        conn.commit()
        stats[r["status"]] += 1
        details.append(
            {
                **r,
                "feedback_id": item["id"],
                "review_id": item.get("review_id"),
                "app": item.get("app"),
                "score": item.get("score"),
                "content": item.get("content"),
            }
        )

    return stats, details
