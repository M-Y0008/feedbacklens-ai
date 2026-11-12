"""界面数据访问层：查询、轻量迁移、上传入库、优先级与导出。

说明：
- 不改动 core/ 的既有逻辑；这里只做 UI 需要的读写编排。
- 迁移只新增表/索引，不触碰 feedback / analyses 既有数据。
"""
from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from core import db as core_db
from core.ingest import _content_hash, normalize_text, read_raw_csv

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = Path(os.environ.get("FEEDBACKLENS_DB", ROOT / "data" / "feedbacklens.db"))
UPLOAD_DIR = ROOT / "data" / "uploads"
EXPORT_DIR = Path(os.environ.get("FEEDBACKLENS_EXPORT_DIR", ROOT / "data" / "exports"))

THEME_LABELS = {
    "account_login": "账号与登录",
    "pricing_subscription": "价格与订阅",
    "answer_quality": "回答质量",
    "speed_latency": "响应速度",
    "stability_crash": "稳定性与崩溃",
    "missing_features": "功能缺失",
    "ui_usability": "界面与易用性",
    "privacy_security": "隐私与安全",
    "customer_support": "客户支持",
    "sync_platform": "跨端同步",
    "model_limits": "模型能力边界",
    "other": "其他（证据不足）",
}
SENTIMENT_LABELS = {
    "positive": "正面", "neutral": "中性", "mixed": "褒贬混合", "negative": "负面",
}
INTENT_LABELS = {
    "praise": "称赞", "complaint": "抱怨", "feature_request": "功能请求",
    "question": "提问", "other": "其他",
}
SEVERITY_LABELS = {
    0: "0 无具体问题", 1: "1 轻微不便", 2: "2 次要功能受影响",
    3: "3 主流程受损", 4: "4 核心体验严重受损", 5: "5 阻断使用/信任受损",
}

SCHEMA_UI = """
CREATE TABLE IF NOT EXISTS priority_weights (
    project_id INTEGER PRIMARY KEY REFERENCES projects(id),
    freq_w REAL NOT NULL DEFAULT 0.30,
    severity_w REAL NOT NULL DEFAULT 0.25,
    impact_w REAL NOT NULL DEFAULT 0.20,
    scope_w REAL NOT NULL DEFAULT 0.15,
    cost_w REAL NOT NULL DEFAULT 0.10,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS export_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL REFERENCES projects(id),
    format TEXT NOT NULL,
    filename TEXT,
    path TEXT,
    rows INTEGER,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS priority_audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL REFERENCES projects(id),
    issue_id INTEGER NOT NULL REFERENCES issues(id),
    actor TEXT,
    field TEXT NOT NULL,
    old_value TEXT,
    new_value TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS eval_label_audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    feedback_id INTEGER NOT NULL REFERENCES feedback(id),
    labeler TEXT NOT NULL,
    field TEXT NOT NULL,
    old_value TEXT,
    new_value TEXT,
    changed_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_feedback_batch ON feedback(batch_id);
"""


def _migrate_ui(conn: sqlite3.Connection) -> None:
    """轻量迁移：只补列/补表，不触碰 feedback 与 analyses 既有数据。"""
    def cols(table: str) -> set[str]:
        return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}

    if "pm_actor" not in cols("priorities"):
        conn.execute("ALTER TABLE priorities ADD COLUMN pm_actor TEXT")
    action_cols = cols("actions")
    for name, ddl in (("priority_tier", "TEXT"), ("model_name", "TEXT"),
                      ("prompt_version", "TEXT"), ("raw_output", "TEXT")):
        if name not in action_cols:
            conn.execute(f"ALTER TABLE actions ADD COLUMN {name} {ddl}")
    if "prompt_version" not in cols("eval_runs"):
        conn.execute("ALTER TABLE eval_runs ADD COLUMN prompt_version TEXT")
    if "sample_size" not in cols("eval_labels"):
        conn.execute("ALTER TABLE eval_labels ADD COLUMN sample_size INTEGER")
    conn.commit()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _ui_migration_needed(conn: sqlite3.Connection) -> bool:
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    for t in ("priority_weights", "export_logs", "priority_audit", "eval_label_audit"):
        if t not in tables:
            return True
    checks = {
        "priorities": {"pm_actor"},
        "actions": {"priority_tier", "model_name", "prompt_version", "raw_output"},
        "eval_runs": {"prompt_version"},
        "eval_labels": {"sample_size"},
    }
    for t, cols in checks.items():
        if t not in tables:
            continue
        have = {r[1] for r in conn.execute(f"PRAGMA table_info({t})")}
        if cols - have:
            return True
    return False


@st.cache_resource
def get_conn() -> sqlite3.Connection:
    path_existed = DB_PATH.exists()
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        conn.execute("PRAGMA journal_mode = WAL")
    except sqlite3.OperationalError as exc:
        raise RuntimeError(
            f"数据库无法以 WAL 打开：{exc}｜path={DB_PATH}｜exists={DB_PATH.exists()}"
            f"｜dir={DB_PATH.parent}｜writable={os.access(DB_PATH.parent, os.W_OK)}"
        ) from exc
    # 首次 UI 迁移前按约定备份数据库（feedbacklens.db.bak），不影响既有数据。
    if path_existed and _ui_migration_needed(conn):
        bak = sqlite3.connect(str(DB_PATH) + ".bak")
        try:
            conn.backup(bak)
        finally:
            bak.close()
    conn.executescript(SCHEMA_UI)
    _migrate_ui(conn)
    return conn


def backup_db(conn: sqlite3.Connection) -> str | None:
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    except sqlite3.Error:
        pass
    bak = core_db.backup_if_exists(DB_PATH)
    return str(bak) if bak else None


def theme_label(code: str) -> str:
    return THEME_LABELS.get(code, code)


# ---------------------------------------------------------------- 项目 / 状态
def list_projects(conn: sqlite3.Connection, include_deleted: bool = False) -> list[dict]:
    where = "" if include_deleted else "WHERE p.deleted_at IS NULL"
    rows = conn.execute(
        f"""
        SELECT p.id, p.name, p.description, p.status, p.created_at, p.updated_at,
               p.deleted_at,
               (SELECT COUNT(*) FROM feedback f WHERE f.project_id=p.id) AS samples,
               (SELECT COUNT(*) FROM products pr WHERE pr.project_id=p.id) AS products
        FROM projects p {where}
        ORDER BY p.id DESC
        """
    ).fetchall()
    return [dict(r) for r in rows]


def project_by_id(conn: sqlite3.Connection, project_id: int) -> dict | None:
    row = conn.execute("SELECT * FROM projects WHERE id=?", (project_id,)).fetchone()
    return dict(row) if row else None


def project_products(conn: sqlite3.Connection, project_id: int) -> list[dict]:
    rows = conn.execute(
        "SELECT id, name, package_name FROM products WHERE project_id=? ORDER BY id",
        (project_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def create_project(conn: sqlite3.Connection, name: str, description: str,
                   products: list[str]) -> int:
    now = utc_now()
    cur = conn.execute(
        "INSERT INTO projects (name, description, status, created_at, updated_at) VALUES (?,?,?,?,?)",
        (name.strip(), description.strip(), "draft", now, now),
    )
    pid = int(cur.lastrowid)
    for p in products:
        if p.strip():
            conn.execute(
                "INSERT OR IGNORE INTO products (project_id, name, package_name) VALUES (?,?,?)",
                (pid, p.strip(), None),
            )
    conn.commit()
    return pid


def update_project_status(conn: sqlite3.Connection, project_id: int, status: str) -> None:
    conn.execute("UPDATE projects SET status=?, updated_at=? WHERE id=?",
                 (status, utc_now(), project_id))
    conn.commit()


def soft_delete_project(conn: sqlite3.Connection, project_id: int) -> None:
    conn.execute("UPDATE projects SET deleted_at=?, updated_at=? WHERE id=?",
                 (utc_now(), utc_now(), project_id))
    conn.commit()


def restore_project(conn: sqlite3.Connection, project_id: int) -> None:
    conn.execute("UPDATE projects SET deleted_at=NULL, updated_at=? WHERE id=?",
                 (utc_now(), project_id))
    conn.commit()


def project_sample_summary(conn: sqlite3.Connection, project_id: int) -> dict:
    row = conn.execute(
        "SELECT COUNT(*) AS samples, MIN(review_at) AS dmin, MAX(review_at) AS dmax "
        "FROM feedback WHERE project_id=?",
        (project_id,),
    ).fetchone()
    return dict(row)


def product_stats(conn: sqlite3.Connection, project_id: int,
                  version: str | None = None) -> list[dict]:
    rows = conn.execute(
        """
        SELECT p.id, p.name,
               COUNT(f.id) AS samples,
               MIN(f.review_at) AS dmin, MAX(f.review_at) AS dmax,
               SUM(CASE WHEN a.status='done' THEN 1 ELSE 0 END) AS done,
               SUM(CASE WHEN a.status='degraded' THEN 1 ELSE 0 END) AS degraded,
               SUM(CASE WHEN a.status='skipped' THEN 1 ELSE 0 END) AS skipped,
               SUM(CASE WHEN a.status='failed' THEN 1 ELSE 0 END) AS failed,
               SUM(CASE WHEN a.needs_human_review=1 THEN 1 ELSE 0 END) AS review_needed
        FROM products p
        LEFT JOIN feedback f ON f.product_id=p.id AND f.project_id=?
        LEFT JOIN analyses a ON a.feedback_id=f.id AND a.prompt_version=?
            AND a.model_name=?
        WHERE p.project_id=?
        GROUP BY p.id, p.name
        ORDER BY p.id
        """,
        (project_id, version, _cfg_model(), project_id),
    ).fetchall()
    return [dict(r) for r in rows]


def version_list(conn: sqlite3.Connection, project_id: int) -> list[str]:
    """当前模型下该项目已出现过的 prompt 版本（按版本号倒序）。"""
    rows = conn.execute(
        "SELECT DISTINCT a.prompt_version FROM analyses a "
        "JOIN feedback f ON f.id=a.feedback_id "
        "WHERE f.project_id=? AND a.model_name=? AND a.prompt_version IS NOT NULL "
        "ORDER BY a.prompt_version DESC",
        (project_id, _cfg_model()),
    ).fetchall()
    return [r[0] for r in rows if r[0]]


def latest_version(conn: sqlite3.Connection, project_id: int) -> str:
    """优先使用配置中的 prompt_version；没有结果时退回到覆盖最多的版本。

    必须按当前模型过滤：analyses 的主键含 model_name，混模型统计会重复计数。
    """
    from core.llm import load_config
    cfg_version = str(load_config().get("prompt_version", "v2.2-p2"))
    model = _cfg_model()
    rows = conn.execute(
        """
        SELECT a.prompt_version, COUNT(*) AS n FROM analyses a
        JOIN feedback f ON f.id=a.feedback_id
        WHERE f.project_id=? AND a.model_name=? AND a.prompt_version IS NOT NULL
          AND a.status IN ('done','degraded','skipped')
        GROUP BY a.prompt_version ORDER BY n DESC, a.prompt_version DESC
        """,
        (project_id, model),
    ).fetchall()
    found = [r[0] for r in rows]
    if cfg_version in found:
        return cfg_version
    return found[0] if found else cfg_version


def status_counts(conn: sqlite3.Connection, project_id: int, version: str) -> dict:
    row = conn.execute(
        """
        SELECT COUNT(*) AS analyzed,
               SUM(CASE WHEN a.status='done' THEN 1 ELSE 0 END) AS done,
               SUM(CASE WHEN a.status='degraded' THEN 1 ELSE 0 END) AS degraded,
               SUM(CASE WHEN a.status='skipped' THEN 1 ELSE 0 END) AS skipped,
               SUM(CASE WHEN a.status='failed' THEN 1 ELSE 0 END) AS failed,
               SUM(CASE WHEN a.needs_human_review=1 THEN 1 ELSE 0 END) AS review_needed
        FROM analyses a JOIN feedback f ON f.id=a.feedback_id
        WHERE f.project_id=? AND a.prompt_version=? AND a.model_name=?
        """,
        (project_id, version, _cfg_model()),
    ).fetchone()
    samples = conn.execute("SELECT COUNT(*) FROM feedback WHERE project_id=?",
                           (project_id,)).fetchone()[0]
    out = {k: int(row[k] or 0) for k in row.keys()}
    out["samples"] = int(samples)
    out["pending"] = int(samples) - int(out["analyzed"])
    return out


def run_history(conn: sqlite3.Connection, project_id: int, limit: int = 12) -> list[dict]:
    rows = conn.execute(
        """
        SELECT a.run_id, a.model_name, a.prompt_version,
               MIN(a.created_at) AS started_at, MAX(a.created_at) AS finished_at,
               COUNT(*) AS rows,
               SUM(CASE WHEN a.status='done' THEN 1 ELSE 0 END) AS done,
               SUM(CASE WHEN a.status='degraded' THEN 1 ELSE 0 END) AS degraded,
               SUM(CASE WHEN a.status='skipped' THEN 1 ELSE 0 END) AS skipped,
               SUM(CASE WHEN a.status='failed' THEN 1 ELSE 0 END) AS failed
        FROM analyses a JOIN feedback f ON f.id=a.feedback_id
        WHERE f.project_id=?
        GROUP BY a.run_id, a.model_name, a.prompt_version
        ORDER BY finished_at DESC LIMIT ?
        """,
        (project_id, limit),
    ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------- 洞察数据
def load_insight_df(conn: sqlite3.Connection, project_id: int, version: str,
                    product: str | None = None) -> pd.DataFrame:
    sql = """
        SELECT f.id AS feedback_id, f.review_id, f.content, f.score, f.review_at,
               p.name AS product, a.themes_json, a.sentiment, a.intent,
               a.pain_point, a.feature_request, a.severity, a.evidence_quote,
               a.confidence, a.needs_human_review, a.status, a.is_valid,
               a.model_name, a.prompt_version
        FROM analyses a
        JOIN feedback f ON f.id=a.feedback_id
        JOIN products p ON p.id=f.product_id
        WHERE f.project_id=? AND a.prompt_version=? AND a.model_name=?
    """
    params: list = [project_id, version, _cfg_model()]
    if product:
        sql += " AND p.name=?"
        params.append(product)
    rows = conn.execute(sql, params).fetchall()
    data = [dict(r) for r in rows]
    df = pd.DataFrame(data)
    if df.empty:
        return df
    df["themes"] = df["themes_json"].map(_parse_themes)
    df["needs_human_review"] = df["needs_human_review"].fillna(0).astype(bool)
    df["severity"] = pd.to_numeric(df["severity"], errors="coerce")
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["review_at"] = pd.to_datetime(df["review_at"], errors="coerce", utc=True)
    df["review_at_str"] = df["review_at"].dt.strftime("%Y-%m-%d").fillna("-")
    df["quote_ok"] = df.apply(_quote_hit, axis=1)
    df["sentiment_cn"] = df["sentiment"].map(SENTIMENT_LABELS).fillna("-")
    df["intent_cn"] = df["intent"].map(INTENT_LABELS).fillna("-")
    return df


def _parse_themes(raw) -> list[str]:
    try:
        t = json.loads(raw or "[]")
        return [str(x) for x in t] if isinstance(t, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


def _quote_hit(row) -> bool:
    q = (row.get("evidence_quote") or "").strip()
    c = normalize_text(row.get("content") or "")
    return bool(q) and normalize_text(q) in c


def theme_counts(df: pd.DataFrame) -> dict:
    """返回 (主题计数, 仅 other 条数, other 出现次数)；主题排行不含 other。"""
    counts: dict[str, int] = {}
    other_only = 0
    other_occ = 0
    for themes in df["themes"]:
        if themes == ["other"]:
            other_only += 1
        for t in themes:
            counts[t] = counts.get(t, 0) + 1
            if t == "other":
                other_occ += 1
    counts.pop("other", None)
    return {"counts": counts, "other_only": other_only, "other_occ": other_occ}


def severity_counts(df: pd.DataFrame) -> dict:
    c = df["severity"].dropna().astype(int).value_counts().sort_index()
    return {int(k): int(v) for k, v in c.items()}


def pain_clusters(df: pd.DataFrame, min_count: int = 1) -> list[dict]:
    return _text_clusters(df, "pain_point", min_count)


def feature_clusters(df: pd.DataFrame, min_count: int = 1) -> list[dict]:
    return _text_clusters(df, "feature_request", min_count)


def _norm_key(text: str) -> str:
    t = normalize_text(str(text)).lower()
    t = re.sub(r"[^\w\u4e00-\u9fff]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _text_clusters(df: pd.DataFrame, col: str, min_count: int) -> list[dict]:
    rows: dict[str, dict] = {}
    for rec in df.to_dict("records"):
        raw = rec.get(col)
        if not raw or not str(raw).strip():
            continue
        key = _norm_key(raw)
        if not key:
            continue
        item = rows.setdefault(key, {
            "label": normalize_text(str(raw)), "count": 0, "ids": [],
            "products": set(), "severity": [], "sentiment": {},
        })
        item["count"] += 1
        item["ids"].append(int(rec["feedback_id"]))
        item["products"].add(rec["product"])
        if pd.notna(rec.get("severity")):
            item["severity"].append(float(rec["severity"]))
        s = rec.get("sentiment") or "unknown"
        item["sentiment"][s] = item["sentiment"].get(s, 0) + 1
    out = []
    for item in rows.values():
        if item["count"] < min_count:
            continue
        out.append({
            "label": item["label"],
            "count": item["count"],
            "ids": item["ids"],
            "products": sorted(item["products"]),
            "avg_severity": round(sum(item["severity"]) / len(item["severity"]), 2)
            if item["severity"] else None,
        })
    out.sort(key=lambda x: (-x["count"], -(x["avg_severity"] or 0)))
    return out


def strengths(df: pd.DataFrame, limit: int = 12) -> list[dict]:
    pos = df[(df["sentiment"] == "positive") | (df["intent"] == "praise")].copy()
    pos = pos[pos["score"].fillna(0) >= 3]
    out = []
    seen = set()
    for rec in pos.sort_values(["score", "review_at"], ascending=[False, False]).to_dict("records"):
        key = _norm_key(rec.get("evidence_quote") or rec["content"][:60])
        if key in seen:
            continue
        seen.add(key)
        themes = [t for t in rec["themes"] if t != "other"]
        out.append({
            "label": normalize_text(rec.get("evidence_quote") or rec["content"])[:160],
            "themes": themes,
            "product": rec["product"],
            "score": rec["score"],
            "feedback_id": int(rec["feedback_id"]),
            "ids": [int(rec["feedback_id"])],
        })
        if len(out) >= limit:
            break
    return out


def evidence_by_ids(df: pd.DataFrame, ids: list[int], limit: int = 500) -> pd.DataFrame:
    if not ids:
        return pd.DataFrame()
    part = df[df["feedback_id"].isin(ids)].copy()
    part["quote"] = part["evidence_quote"]
    part = part.sort_values(["severity", "score"], ascending=[False, True]).head(limit)
    cols = ["feedback_id", "review_id", "product", "score", "review_at_str",
            "severity", "sentiment_cn", "needs_human_review", "quote", "content"]
    out = part[cols].rename(columns={"review_at_str": "review_at",
                                     "sentiment_cn": "sentiment"})
    return out


# ---------------------------------------------------------------- CSV 上传
FIELD_ALIASES = {
    "review_id": ["review_id", "reviewid", "id", "评论id", "review id"],
    "content": ["content", "review", "review_text", "text", "body", "评论", "评论内容", "正文"],
    "score": ["score", "rating", "stars", "star", "评分", "星级"],
    "at_utc": ["at_utc", "date", "review_date", "review_at", "time", "created_at",
               "评论时间", "日期", "时间"],
    "app_name": ["app_name", "app", "product", "application", "应用", "产品"],
    "package_name": ["package_name", "package", "bundle", "包名"],
}


def detect_mapping(columns: list[str]) -> dict:
    lower = {str(c).strip().lower(): c for c in columns}
    mapping = {}
    for field, aliases in FIELD_ALIASES.items():
        for a in aliases:
            if a in lower:
                mapping[field] = lower[a]
                break
    return mapping


def save_upload(uploaded) -> Path:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^\w.\-]+", "_", uploaded.name)
    path = UPLOAD_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe}"
    path.write_bytes(uploaded.getvalue())
    return path


def read_uploaded_csv(path: Path) -> pd.DataFrame:
    return read_raw_csv(path)


def validate_upload(raw: pd.DataFrame, mapping: dict, project_products: list[str],
                    existing_review_ids: set[str],
                    allow_new_products: bool = False) -> dict:
    """规范化 + 逐行校验，返回 accepted / rejected / quality。"""
    missing = [k for k in ("review_id", "content", "score", "app_name") if k not in mapping]
    if missing:
        return {"ok": False, "missing_fields": missing, "accepted": pd.DataFrame(),
                "rejected": pd.DataFrame(), "quality": {}}

    norm = pd.DataFrame({
        "review_id": raw[mapping["review_id"]].astype(str).str.strip(),
        "content": raw[mapping["content"]].astype(str).map(normalize_text),
        "score_raw": raw[mapping["score"]].astype(str).str.strip(),
        "app_name": raw[mapping["app_name"]].astype(str).str.strip(),
        "package_name": raw[mapping["package_name"]].astype(str).str.strip()
        if "package_name" in mapping else "",
        "at_raw": raw[mapping["at_utc"]].astype(str).str.strip()
        if "at_utc" in mapping else "",
    })
    norm["score"] = pd.to_numeric(norm["score_raw"], errors="coerce")
    dates = pd.to_datetime(norm["at_raw"], errors="coerce", utc=True)
    norm["review_at"] = dates.map(lambda x: x.isoformat() if pd.notna(x) else None)

    blank_id = norm["review_id"].str.lower().isin(["", "nan", "none"])
    for i in norm.index[blank_id]:
        norm.at[i, "review_id"] = f"auto-{i}-{uuid.uuid4().hex[:8]}"

    reasons: list[str] = []
    flags: list[str] = []
    for i, r in norm.iterrows():
        rs, fl = [], []
        if not r["content"] or r["content"].lower() in ("nan", "none"):
            rs.append("正文为空")
        elif len(r["content"]) < 5:
            rs.append("正文过短(<5字符)")
        if pd.isna(r["score"]):
            rs.append("评分为空或非数字")
        elif int(r["score"]) not in (1, 2, 3, 4, 5):
            rs.append("评分越界(非1-5)")
        if not r["app_name"] or r["app_name"].lower() in ("nan", "none"):
            rs.append("应用名为空")
        elif r["app_name"] not in project_products and not allow_new_products:
            rs.append("应用不在项目产品列表")
        if not r["at_raw"] or r["at_raw"].lower() in ("nan", "none"):
            fl.append("日期缺失")
        elif pd.isna(dates.loc[i]):
            fl.append("日期无法解析")
        if len(r["content"]) > 2000:
            fl.append("正文超长(>2000字符)")
        if any(p.search(r["content"]) for p in
               (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), re.compile(r"https?://\S+"))):
            fl.append("含联系方式/链接")
        if r["review_id"] in existing_review_ids:
            rs.append("库里已存在同一 review_id")
        reasons.append("；".join(dict.fromkeys(rs)))
        flags.append("；".join(dict.fromkeys(fl)))

    norm["reason"] = reasons
    norm["flags"] = flags

    dup_id = norm.loc[norm["review_id"].duplicated(keep="first")].index
    for i in dup_id:
        norm.at[i, "reason"] = "；".join(filter(None, [norm.at[i, "reason"], "文件内重复 review_id"]))
    dup_content = norm.loc[norm.duplicated(subset=["content", "app_name"], keep="first")].index
    for i in dup_content:
        norm.at[i, "reason"] = "；".join(filter(None, [norm.at[i, "reason"], "文件内重复正文+应用"]))

    accepted_mask = norm["reason"] == ""
    accepted = norm[accepted_mask].copy()
    rejected = norm[~accepted_mask].copy()

    reason_counts: dict[str, int] = {}
    for r in rejected["reason"]:
        for part in [x for x in r.split("；") if x]:
            reason_counts[part] = reason_counts.get(part, 0) + 1

    quality = {
        "rows": int(len(norm)),
        "accepted": int(len(accepted)),
        "rejected": int(len(rejected)),
        "reason_counts": reason_counts,
        "missing": {f: int(_raw_missing(raw, mapping.get(f))) for f in mapping if mapping.get(f)},
        "duplicate_review_id": int(norm["review_id"].duplicated().sum()),
        "duplicate_content": int(norm.duplicated(subset=["content", "app_name"]).sum()),
        "score_counts": {str(k): int(v) for k, v in
                         norm["score"].dropna().astype(int).value_counts().sort_index().items()},
        "score_out_of_range": int((~norm["score"].isin([1, 2, 3, 4, 5]) & norm["score"].notna()).sum()),
        "date_parse_fail": int(sum(1 for f in flags if "日期无法解析" in f)),
        "date_min": str(norm["review_at"].dropna().min()) if norm["review_at"].notna().any() else "-",
        "date_max": str(norm["review_at"].dropna().max()) if norm["review_at"].notna().any() else "-",
        "short_lt5": int((norm["content"].str.len() < 5).sum()),
        "short_lt10": int((norm["content"].str.len() < 10).sum()),
        "content_len_median": float(norm["content"].str.len().median()) if len(norm) else 0,
        "sensitive_hits": int(sum(1 for f in flags if "含联系方式" in f)),
        "app_counts": {str(k): int(v) for k, v in norm["app_name"].value_counts().items()},
        "unknown_apps": sorted({a for a in norm["app_name"].unique()
                                if a and a.lower() != "nan" and a not in project_products}),
        "week_counts": {},
    }
    if len(norm) and "at_utc" in mapping:
        wk = dates.dropna().dt.tz_localize(None).dt.to_period("W").astype(str).value_counts().sort_index()
        quality["week_counts"] = {str(k): int(v) for k, v in wk.items()}
    return {"ok": True, "accepted": accepted, "rejected": rejected, "quality": quality,
            "mapping": mapping}


def _raw_missing(raw: pd.DataFrame, col: str | None) -> int:
    if not col or col not in raw.columns:
        return 0
    s = raw[col].astype(str).str.strip()
    return int(((s == "") | (s.str.lower() == "nan")).sum())


def commit_import(conn: sqlite3.Connection, project_id: int, accepted: pd.DataFrame,
                  filename: str, quality: dict,
                  new_products: list[str] | None = None) -> dict:
    batch_id = str(uuid.uuid4())
    now = utc_now()
    backup = backup_db(conn)

    product_ids = {p["name"]: p["id"] for p in project_products(conn, project_id)}
    for name in new_products or []:
        if name not in product_ids:
            cur = conn.execute(
                "INSERT INTO products (project_id, name, package_name) VALUES (?,?,?)",
                (project_id, name, None),
            )
            product_ids[name] = int(cur.lastrowid)

    conn.execute(
        "INSERT INTO import_batches (id, project_id, filename, row_count, accepted_count, "
        "rejected_count, quality_report_json, created_at) VALUES (?,?,?,?,?,?,?,?)",
        (batch_id, project_id, filename, int(quality.get("rows") or len(accepted)),
         int(len(accepted)), int(quality.get("rejected", 0)),
         json.dumps(quality, ensure_ascii=False), now),
    )
    inserted = 0
    for r in accepted.itertuples(index=False):
        pid = product_ids.get(str(r.app_name))
        if pid is None:
            continue
        cur = conn.execute(
            "INSERT OR IGNORE INTO feedback (project_id, batch_id, review_id, product_id, "
            "content, score, review_at, lang, source, meta_json, content_hash, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (project_id, batch_id, str(r.review_id), pid, str(r.content),
             int(float(r.score)), r.review_at, "auto", "ui_upload", "{}",
             _content_hash(str(r.content)), now),
        )
        inserted += int(cur.rowcount or 0)
    conn.execute("UPDATE projects SET updated_at=? WHERE id=?", (now, project_id))
    conn.commit()
    return {"batch_id": batch_id, "inserted": inserted,
            "skipped_existing": int(len(accepted)) - inserted, "backup": backup}


def list_batches(conn: sqlite3.Connection, project_id: int) -> list[dict]:
    rows = conn.execute(
        """
        SELECT b.id, b.filename, b.row_count, b.accepted_count, b.rejected_count, b.created_at,
               (SELECT COUNT(*) FROM feedback f WHERE f.batch_id=b.id) AS stored,
               (SELECT COUNT(*) FROM analyses a JOIN feedback f ON f.id=a.feedback_id
                WHERE f.batch_id=b.id) AS analyzed
        FROM import_batches b WHERE b.project_id=? ORDER BY b.created_at DESC
        """,
        (project_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def rollback_batch(conn: sqlite3.Connection, batch_id: str) -> dict:
    analyzed = conn.execute(
        "SELECT COUNT(*) FROM analyses a JOIN feedback f ON f.id=a.feedback_id WHERE f.batch_id=?",
        (batch_id,),
    ).fetchone()[0]
    if analyzed:
        return {"ok": False, "reason": f"该批次已有 {analyzed} 条分析结果，不能回滚"}
    backup = backup_db(conn)
    n = conn.execute("DELETE FROM feedback WHERE batch_id=?", (batch_id,)).rowcount
    conn.execute("DELETE FROM import_batches WHERE id=?", (batch_id,))
    conn.commit()
    return {"ok": True, "deleted": int(n), "backup": backup}


# ---------------------------------------------------------------- AI 分析
def pending_items(conn: sqlite3.Connection, project_id: int, version: str,
                  product: str | None = None, limit: int | None = None) -> list[dict]:
    sql = (
        "SELECT f.id, f.review_id, f.content, f.score, p.name AS app "
        "FROM feedback f JOIN products p ON p.id=f.product_id "
        "WHERE f.project_id=? AND NOT EXISTS ("
        "  SELECT 1 FROM analyses a WHERE a.feedback_id=f.id "
        "  AND a.model_name=? AND a.prompt_version=?"
        ") ORDER BY f.id"
    )
    model = _cfg_model()
    params: list = [project_id, model, version]
    if product:
        sql = sql.replace("ORDER BY f.id", "AND p.name=? ORDER BY f.id")
        params.append(product)
    if limit:
        sql += " LIMIT ?"
        params.append(int(limit))
    rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def rerun_items(conn: sqlite3.Connection, project_id: int, version: str,
                statuses: tuple[str, ...], limit: int) -> list[dict]:
    sql = (
        "SELECT f.id, f.review_id, f.content, f.score, p.name AS app "
        "FROM analyses a JOIN feedback f ON f.id=a.feedback_id "
        "JOIN products p ON p.id=f.product_id "
        "WHERE f.project_id=? AND a.prompt_version=? AND a.status IN ({}) "
        "ORDER BY a.created_at DESC LIMIT ?"
    ).format(",".join("?" for _ in statuses))
    rows = conn.execute(sql, (project_id, version, *statuses, int(limit))).fetchall()
    return [dict(r) for r in rows]


def _cfg_model() -> str:
    from core.llm import load_config
    cfg = load_config()
    return str(cfg.get("backends", {}).get("ollama", {}).get("model", "qwen3:8b"))


def analysis_summary_rows(conn: sqlite3.Connection, project_id: int, version: str) -> pd.DataFrame:
    rows = conn.execute(
        """
        SELECT p.name AS 产品,
               COUNT(f.id) AS 样本量,
               SUM(CASE WHEN a.status='done' THEN 1 ELSE 0 END) AS done,
               SUM(CASE WHEN a.status='degraded' THEN 1 ELSE 0 END) AS degraded,
               SUM(CASE WHEN a.status='skipped' THEN 1 ELSE 0 END) AS skipped,
               SUM(CASE WHEN a.status='failed' THEN 1 ELSE 0 END) AS failed,
               SUM(CASE WHEN a.needs_human_review=1 THEN 1 ELSE 0 END) AS 需人工复核
        FROM products p
        LEFT JOIN feedback f ON f.product_id=p.id AND f.project_id=?
        LEFT JOIN analyses a ON a.feedback_id=f.id AND a.prompt_version=?
            AND a.model_name=?
        WHERE p.project_id=? GROUP BY p.id, p.name ORDER BY p.id
        """,
        (project_id, version, _cfg_model(), project_id),
    ).fetchall()
    return pd.DataFrame([dict(r) for r in rows])


# ---------------------------------------------------------------- 优先级
DEFAULT_WEIGHTS = {"freq": 0.30, "severity": 0.25, "impact": 0.20, "scope": 0.15, "cost": 0.10}


def get_weights(conn: sqlite3.Connection, project_id: int) -> dict:
    row = conn.execute("SELECT * FROM priority_weights WHERE project_id=?", (project_id,)).fetchone()
    if not row:
        return dict(DEFAULT_WEIGHTS)
    return {"freq": row["freq_w"], "severity": row["severity_w"], "impact": row["impact_w"],
            "scope": row["scope_w"], "cost": row["cost_w"]}


def save_weights(conn: sqlite3.Connection, project_id: int, w: dict) -> None:
    conn.execute(
        "INSERT INTO priority_weights (project_id, freq_w, severity_w, impact_w, scope_w, cost_w, updated_at) "
        "VALUES (?,?,?,?,?,?,?) ON CONFLICT(project_id) DO UPDATE SET "
        "freq_w=excluded.freq_w, severity_w=excluded.severity_w, impact_w=excluded.impact_w, "
        "scope_w=excluded.scope_w, cost_w=excluded.cost_w, updated_at=excluded.updated_at",
        (project_id, w["freq"], w["severity"], w["impact"], w["scope"], w["cost"], utc_now()),
    )
    conn.commit()


def sync_issue_candidates(conn: sqlite3.Connection, project_id: int,
                          version: str) -> list[dict]:
    """按主题把真实证据聚成候选问题（other 不参与），写入 issues 表。"""
    df = load_insight_df(conn, project_id, version)
    if df.empty:
        return []
    counts = theme_counts(df)["counts"]
    now = utc_now()
    existing = {r["theme"]: dict(r) for r in conn.execute(
        "SELECT * FROM issues WHERE project_id=?", (project_id,)).fetchall()}
    out = []
    for theme, n in sorted(counts.items(), key=lambda x: -x[1]):
        if n < 3:
            continue
        ev = df[df["themes"].map(lambda t: theme in t)]
        ids = [int(x) for x in ev["feedback_id"].tolist()]
        products = {p: int(c) for p, c in ev["product"].value_counts().items()}
        avg_sev = ev["severity"].dropna()
        title = theme_label(theme)
        payload = (title, theme, json.dumps(ids), len(ids),
                   json.dumps(products, ensure_ascii=False))
        if theme in existing:
            issue_id = existing[theme]["id"]
            conn.execute(
                "UPDATE issues SET title=?, evidence_ids_json=?, evidence_count=?, products_json=? "
                "WHERE id=?", (title, payload[2], payload[3], payload[4], issue_id))
        else:
            cur = conn.execute(
                "INSERT INTO issues (project_id, title, theme, evidence_ids_json, evidence_count, "
                "products_json, status, created_at) VALUES (?,?,?,?,?,?,?,?)",
                (project_id, title, theme, payload[2], payload[3], payload[4], "candidate", now))
            issue_id = int(cur.lastrowid)
        out.append({
            "issue_id": issue_id, "title": title, "theme": theme, "count": len(ids),
            "ids": ids, "products": products,
            "avg_severity": round(float(avg_sev.mean()), 2) if len(avg_sev) else None,
        })
    conn.commit()
    return out


def _freq_score(count: int, max_count: int) -> float:
    if max_count <= 1:
        return 1.0
    return round(5.0 * math.log(1 + count) / math.log(1 + max_count), 2)


def compute_tier(total: float) -> str:
    if total >= 4.0:
        return "P1"
    if total >= 2.5:
        return "P2"
    return "P3"


def priority_rows(conn: sqlite3.Connection, project_id: int, version: str) -> list[dict]:
    """返回候选问题 + 已保存的人工/成本设置（未保存的给默认值，不写库）。"""
    issues = sync_issue_candidates(conn, project_id, version)
    if not issues:
        return []
    max_count = max(i["count"] for i in issues)
    saved = {r["issue_id"]: dict(r) for r in conn.execute(
        "SELECT * FROM priorities WHERE project_id=?", (project_id,)).fetchall()}
    w = get_weights(conn, project_id)
    out = []
    for it in issues:
        s = saved.get(it["issue_id"], {})
        cost = float(s.get("cost_score") if s.get("cost_score") is not None else 3)
        impact = float(s.get("impact_score") if s.get("impact_score") is not None else
                       (it["avg_severity"] if it["avg_severity"] is not None else 3))
        scope = float(s.get("scope_score") if s.get("scope_score") is not None else
                      min(5.0, 1.0 + 2.0 * (len(it["products"]) - 1)))
        freq = _freq_score(it["count"], max_count)
        sev = float(it["avg_severity"] or 0)
        total = round(w["freq"] * freq + w["severity"] * sev + w["impact"] * impact
                      + w["scope"] * scope - w["cost"] * cost, 2)
        ai_tier = compute_tier(total)
        out.append({
            **it, "freq_score": freq, "severity_score": sev, "impact_score": impact,
            "scope_score": scope, "cost_score": cost, "total_score": total,
            "ai_tier": ai_tier,
            "pm_override_tier": s.get("pm_override_tier"),
            "pm_note": s.get("pm_note"),
            "pm_actor": s.get("pm_actor"),
            "rationale": s.get("rationale"),
            "saved": bool(s),
        })
    out.sort(key=lambda x: -x["total_score"])
    return out


def score_total(row: dict, weights: dict) -> float:
    """按当前权重重算总分：频率 0.30 / 严重度 0.25 / 核心体验 0.20 / 影响范围 0.15 − 成本 0.10。"""
    return round(
        weights["freq"] * float(row["freq_score"])
        + weights["severity"] * float(row["severity_score"])
        + weights["impact"] * float(row["impact_score"])
        + weights["scope"] * float(row["scope_score"])
        - weights["cost"] * float(row["cost_score"]),
        2,
    )


def save_priority(conn: sqlite3.Connection, project_id: int, row: dict,
                  pm_tier: str | None, pm_note: str, rationale: str,
                  actor: str = "PM") -> None:
    """保存优先级设置；人工改判/成本变更/权重结果变化都写入审计表。"""
    now = utc_now()
    exist = conn.execute(
        "SELECT * FROM priorities WHERE project_id=? AND issue_id=?",
        (project_id, row["issue_id"]),
    ).fetchone()

    def audit(field: str, old, new) -> None:
        if str(old) == str(new):
            return
        conn.execute(
            "INSERT INTO priority_audit (project_id, issue_id, actor, field, old_value, "
            "new_value, created_at) VALUES (?,?,?,?,?,?,?)",
            (project_id, row["issue_id"], actor.strip() or "PM", field,
             None if old is None else str(old), None if new is None else str(new), now),
        )

    if exist:
        new_vals = {
            "成本": row["cost_score"],
            "核心体验影响": row["impact_score"],
            "影响范围": row["scope_score"],
            "总分": row["total_score"],
            "AI 分级": row["ai_tier"],
            "人工分级": pm_tier,
            "判断理由": rationale,
        }
        for field, col in (("成本", "cost_score"), ("核心体验影响", "impact_score"),
                           ("影响范围", "scope_score"), ("总分", "total_score"),
                           ("AI 分级", "ai_tier"), ("人工分级", "pm_override_tier"),
                           ("判断理由", "rationale")):
            audit(field, exist[col], new_vals[field])
        conn.execute(
            "UPDATE priorities SET freq_score=?, severity_score=?, impact_score=?, scope_score=?, "
            "cost_score=?, total_score=?, ai_tier=?, pm_override_tier=?, rationale=?, pm_note=?, "
            "pm_actor=?, updated_at=? WHERE id=?",
            (row["freq_score"], row["severity_score"], row["impact_score"], row["scope_score"],
             row["cost_score"], row["total_score"], row["ai_tier"], pm_tier, rationale,
             pm_note, actor.strip() or "PM", now, exist["id"]),
        )
    else:
        conn.execute(
            "INSERT INTO priorities (project_id, issue_id, freq_score, severity_score, impact_score, "
            "scope_score, cost_score, total_score, ai_tier, pm_override_tier, rationale, pm_note, "
            "pm_actor, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (project_id, row["issue_id"], row["freq_score"], row["severity_score"],
             row["impact_score"], row["scope_score"], row["cost_score"], row["total_score"],
             row["ai_tier"], pm_tier, rationale, pm_note, actor.strip() or "PM", now),
        )
        conn.execute(
            "INSERT INTO priority_audit (project_id, issue_id, actor, field, old_value, "
            "new_value, created_at) VALUES (?,?,?,?,?,?,?)",
            (project_id, row["issue_id"], actor.strip() or "PM", "初始评估", None,
             f"总分 {row['total_score']} / {row['ai_tier']}", now),
        )
    conn.commit()


def list_priority_audit(conn: sqlite3.Connection, project_id: int,
                        limit: int = 60) -> list[dict]:
    rows = conn.execute(
        "SELECT a.*, i.title FROM priority_audit a LEFT JOIN issues i ON i.id=a.issue_id "
        "WHERE a.project_id=? ORDER BY a.id DESC LIMIT ?",
        (project_id, limit),
    ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------- 改进方案
def issue_evidence(conn: sqlite3.Connection, project_id: int, version: str,
                   issue: dict, limit: int = 8) -> pd.DataFrame:
    df = load_insight_df(conn, project_id, version)
    return evidence_by_ids(df, issue["ids"], limit=limit)


def list_actions(conn: sqlite3.Connection, project_id: int) -> dict:
    rows = conn.execute(
        "SELECT a.*, i.title, i.theme FROM actions a JOIN issues i ON i.id=a.issue_id "
        "WHERE a.project_id=? ORDER BY a.id DESC", (project_id,)).fetchall()
    return {r["issue_id"]: dict(r) for r in rows}


def _join_text(value) -> str:
    """模型可能返回字符串或列表；统一成中文可读文本。"""
    if isinstance(value, list):
        return "；".join(str(x).strip() for x in value if str(x).strip())
    return str(value or "").strip()


def generate_action(backend, issue: dict, quotes: list[str]) -> dict:
    """用本地模型基于证据生成改进建议草案（标注 AI 建议、未经验证）。"""
    system = (
        "You are a product manager assistant. Based ONLY on the user reviews provided, "
        "draft one improvement proposal. Return ONLY JSON with keys: "
        '{"proposal": "...", "expected_effect": "...", "assumptions": "...", "uncertainty": "..."}. '
        "Write all field values in Simplified Chinese. "
        "Rules: do not invent statistics or user numbers; keep proposal concrete and bounded; "
        "use strings (not arrays) for assumptions and uncertainty, separated by 分号; "
        "explicitly list what is unverified in uncertainty; 2-4 sentences per field."
    )
    def build_user(qs: list[str]) -> str:
        return (
            f"Issue: {issue['title']} (theme: {issue['theme']}, evidence_count: {issue['count']})\n"
            "Evidence quotes:\n" + "\n".join(f"- {q}" for q in qs)
        )

    def parse(text: str) -> dict:
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            return {"proposal": text[:800], "expected_effect": "", "assumptions": "",
                    "uncertainty": "模型输出非严格 JSON，需人工复核"}
        return obj if isinstance(obj, dict) else {}

    # 本地模型对个别引文组合偶发返回 {}；依次减少引文重试，仍失败则明确报错，不写空草案。
    resp, obj = None, {}
    for qs in (quotes[:5], quotes[:3], quotes[:8]):
        resp = backend.chat_json(system, build_user(qs))
        obj = parse(resp.text)
        if _join_text(obj.get("proposal")):
            break
    proposal = _join_text(obj.get("proposal"))
    if not proposal or resp is None:
        raise ValueError("本地模型未返回有效草案（可能显存不足或超时），请重试或改用人工补充。")
    return {
        "proposal": proposal,
        "expected_effect": _join_text(obj.get("expected_effect")),
        "assumptions": _join_text(obj.get("assumptions")),
        "uncertainty": _join_text(obj.get("uncertainty")) or "AI 建议，未经验证",
        "model": getattr(backend, "model", ""), "raw": resp.text,
    }


def save_action(conn: sqlite3.Connection, project_id: int, issue_id: int, draft: dict,
                priority_tier: str | None = None) -> None:
    snap = model_config_snapshot()
    exist = conn.execute(
        "SELECT id FROM actions WHERE project_id=? AND issue_id=?", (project_id, issue_id)
    ).fetchone()
    if exist:
        conn.execute(
            "UPDATE actions SET proposal=?, expected_effect=?, assumptions=?, uncertainty=?, "
            "priority_tier=?, model_name=?, prompt_version=?, raw_output=?, status='ai_draft', "
            "decided_by=NULL, decided_at=NULL, reject_reason=NULL WHERE id=?",
            (draft["proposal"], draft["expected_effect"], draft["assumptions"],
             draft["uncertainty"], priority_tier, draft.get("model") or snap["model"],
             snap["prompt_version"], draft.get("raw"), exist["id"]),
        )
    else:
        conn.execute(
            "INSERT INTO actions (project_id, issue_id, proposal, expected_effect, assumptions, "
            "uncertainty, priority_tier, model_name, prompt_version, raw_output, status) "
            "VALUES (?,?,?,?,?,?,?,?,?,?, 'ai_draft')",
            (project_id, issue_id, draft["proposal"], draft["expected_effect"],
             draft["assumptions"], draft["uncertainty"], priority_tier,
             draft.get("model") or snap["model"], snap["prompt_version"], draft.get("raw")),
        )
    conn.commit()


def decide_action(conn: sqlite3.Connection, action_id: int, status: str,
                  decided_by: str, reason: str = "") -> None:
    conn.execute(
        "UPDATE actions SET status=?, decided_by=?, decided_at=?, reject_reason=? WHERE id=?",
        (status, decided_by.strip() or "PM", utc_now(), reason.strip() or None, action_id),
    )
    conn.commit()


# ---------------------------------------------------------------- 质量评估
def sample_for_labeling(conn: sqlite3.Connection, project_id: int, version: str,
                        per_product: int = 10) -> pd.DataFrame:
    df = load_insight_df(conn, project_id, version)
    if df.empty:
        return df
    parts = []
    for _, g in df.groupby("product"):
        g = g.sort_values(["score", "feedback_id"])
        n = len(g)
        if n <= per_product:
            idx = list(range(n))
        elif per_product == 1:
            idx = [n // 2]
        else:
            idx = sorted({round(i * (n - 1) / (per_product - 1)) for i in range(per_product)})
        parts.append(g.iloc[idx])
    return pd.concat(parts, ignore_index=True)


def list_labels(conn: sqlite3.Connection, project_id: int, version: str,
                labeler: str) -> pd.DataFrame:
    rows = conn.execute(
        """
        SELECT l.feedback_id, l.gold_themes_json, l.gold_sentiment, l.gold_intent,
               l.gold_severity, l.notes, l.labeled_at, l.labeler, l.sample_size
        FROM eval_labels l JOIN feedback f ON f.id=l.feedback_id
        WHERE f.project_id=? AND l.labeler=?
        """,
        (project_id, labeler),
    ).fetchall()
    labels = pd.DataFrame([dict(r) for r in rows])
    if labels.empty:
        return labels
    ai = load_insight_df(conn, project_id, version)
    ai = ai[ai["feedback_id"].isin(labels["feedback_id"])][
        ["feedback_id", "themes", "sentiment", "intent", "severity", "product",
         "score", "quote_ok", "needs_human_review", "content", "review_at_str"]
    ]
    out = labels.merge(ai, on="feedback_id", how="left")
    out["gold_themes"] = out["gold_themes_json"].map(_parse_themes)
    return out


def save_label(conn: sqlite3.Connection, feedback_id: int, labeler: str,
               themes: list[str], sentiment: str, intent: str, severity: int | None,
               notes: str, sample_size: int | None = None) -> bool:
    """保存人工标注；覆盖已有记录时逐字段写审计，返回是否为覆盖更新。"""
    now = utc_now()
    exist = conn.execute(
        "SELECT gold_themes_json, gold_sentiment, gold_intent, gold_severity, notes "
        "FROM eval_labels WHERE feedback_id=? AND labeler=?",
        (feedback_id, labeler),
    ).fetchone()
    themes_json = json.dumps(themes, ensure_ascii=False)
    conn.execute(
        "INSERT INTO eval_labels (feedback_id, labeler, gold_themes_json, gold_sentiment, "
        "gold_intent, gold_severity, notes, labeled_at, sample_size) VALUES (?,?,?,?,?,?,?,?,?) "
        "ON CONFLICT(feedback_id, labeler) DO UPDATE SET "
        "gold_themes_json=excluded.gold_themes_json, "
        "gold_sentiment=excluded.gold_sentiment, gold_intent=excluded.gold_intent, "
        "gold_severity=excluded.gold_severity, notes=excluded.notes, labeled_at=excluded.labeled_at, "
        "sample_size=COALESCE(eval_labels.sample_size, excluded.sample_size)",
        (feedback_id, labeler, themes_json, sentiment, intent, severity, notes, now, sample_size),
    )
    if exist:
        changes = (
            ("主题", exist["gold_themes_json"], themes_json),
            ("情感", exist["gold_sentiment"], sentiment),
            ("意图", exist["gold_intent"], intent),
            ("严重度", exist["gold_severity"], severity),
            ("备注", exist["notes"], notes),
        )
        for field, old, new in changes:
            if str(old) == str(new):
                continue
            conn.execute(
                "INSERT INTO eval_label_audit (feedback_id, labeler, field, old_value, "
                "new_value, changed_at) VALUES (?,?,?,?,?,?)",
                (feedback_id, labeler, field, None if old is None else str(old),
                 None if new is None else str(new), now),
            )
    conn.commit()
    return exist is not None


def list_label_audit(conn: sqlite3.Connection, project_id: int, labeler: str,
                     limit: int = 100) -> pd.DataFrame:
    rows = conn.execute(
        "SELECT a.feedback_id, a.field, a.old_value, a.new_value, a.changed_at, "
        "p.name AS product, substr(f.content, 1, 40) AS content "
        "FROM eval_label_audit a JOIN feedback f ON f.id=a.feedback_id "
        "LEFT JOIN products p ON p.id=f.product_id "
        "WHERE f.project_id=? AND a.labeler=? ORDER BY a.id DESC LIMIT ?",
        (project_id, labeler, limit),
    ).fetchall()
    return pd.DataFrame([dict(r) for r in rows])


def compute_label_metrics(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    n = len(df)
    theme_hit = df.apply(lambda r: bool(set(r["gold_themes"]) & set(r["themes"] or [])), axis=1)
    theme_exact = df.apply(lambda r: set(r["gold_themes"]) == set(r["themes"] or []), axis=1)
    sent_ok = df["gold_sentiment"] == df["sentiment"]
    intent_ok = df["gold_intent"] == df["intent"]
    sev_ok = pd.Series([_sev_close(a, b) for a, b in zip(df["gold_severity"], df["severity"])])
    all_ok = theme_exact & sent_ok & intent_ok & sev_ok
    return {
        "labeled": n,
        "theme_hit_rate": round(float(theme_hit.mean()) * 100, 1),
        "theme_exact_rate": round(float(theme_exact.mean()) * 100, 1),
        "sentiment_acc": round(float(sent_ok.mean()) * 100, 1),
        "intent_acc": round(float(intent_ok.mean()) * 100, 1),
        "severity_within1": round(float(sev_ok.mean()) * 100, 1),
        "all_fields_exact": round(float(all_ok.mean()) * 100, 1),
        "quote_hit_rate": round(float(df["quote_ok"].fillna(False).mean()) * 100, 1),
    }


def _sev_close(gold, ai) -> bool:
    if pd.isna(gold) or pd.isna(ai):
        return False
    return abs(float(gold) - float(ai)) <= 1


def sentiment_confusion(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    return pd.crosstab(df["gold_sentiment"], df["sentiment"], dropna=False)


def save_eval_run(conn: sqlite3.Connection, run_id: str, sample_size: int, labeler: str,
                  metrics: dict, confusion: pd.DataFrame, model_name: str,
                  prompt_version: str | None = None) -> None:
    conn.execute(
        "INSERT INTO eval_runs (run_id, sample_size, labeler, metrics_json, confusion_json, "
        "model_name, prompt_version, created_at) VALUES (?,?,?,?,?,?,?,?)",
        (run_id, int(sample_size), labeler, json.dumps(metrics, ensure_ascii=False),
         confusion.to_json(force_ascii=False) if not confusion.empty else None,
         model_name, prompt_version, utc_now()),
    )
    conn.commit()


def list_eval_runs(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM eval_runs ORDER BY id DESC LIMIT 20").fetchall()
    return [dict(r) for r in rows]


def model_config_snapshot() -> dict:
    from core.llm import load_config
    cfg = load_config()
    b = cfg.get("backends", {}).get("ollama", {})
    return {
        "active_backend": cfg.get("active_backend"),
        "model": b.get("model"),
        "prompt_version": cfg.get("prompt_version"),
        "temperature": b.get("temperature"),
        "num_ctx": b.get("num_ctx"),
        "timeout_seconds": b.get("timeout_seconds"),
    }


# ---------------------------------------------------------------- 导出
def log_export(conn: sqlite3.Connection, project_id: int, fmt: str,
               filename: str, path: str, rows: int) -> None:
    conn.execute(
        "INSERT INTO export_logs (project_id, format, filename, path, rows, created_at) "
        "VALUES (?,?,?,?,?,?)",
        (project_id, fmt, filename, path, int(rows), utc_now()),
    )
    conn.commit()


def list_exports(conn: sqlite3.Connection, project_id: int) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM export_logs WHERE project_id=? ORDER BY id DESC", (project_id,)
    ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------- 竞品对比
def product_compare_rows(df: pd.DataFrame) -> list[dict]:
    """按产品汇总主题/情感/严重度差异，并给出真实样本窗（不做跨期外推）。"""
    out = []
    for name, g in df.groupby("product"):
        tc = theme_counts(g)
        sent = g["sentiment"].value_counts()
        sev = g["severity"].dropna()
        days = (g["review_at"].max() - g["review_at"].min()).days + 1 \
            if g["review_at"].notna().any() else 0
        top = sorted(tc["counts"].items(), key=lambda x: -x[1])[:5]
        out.append({
            "product": name,
            "samples": len(g),
            "dmin": g["review_at"].min(),
            "dmax": g["review_at"].max(),
            "days": days,
            "top_themes": [(theme_label(k), v) for k, v in top],
            "theme_counts": tc["counts"],
            "sentiment": {k: int(v) for k, v in sent.items()},
            "sentiment_pct": {k: round(v / max(1, len(g)) * 100, 1)
                              for k, v in sent.items()},
            "avg_severity": round(float(sev.mean()), 2) if len(sev) else None,
            "sev4_plus": int((sev >= 4).sum()),
            "other_only": tc["other_only"],
            "quote_hit": round(float(g["quote_ok"].mean()) * 100, 1),
        })
    out.sort(key=lambda x: x["product"])
    return out


# ---------------------------------------------------------------- 报告导出
def _fmt_dt(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    try:
        return pd.to_datetime(value, utc=True).strftime("%Y-%m-%d %H:%M")
    except (ValueError, TypeError):
        return str(value)


def build_report_frames(conn: sqlite3.Connection, project_id: int, version: str) -> dict:
    """生成导出用的九张表（全部来自真实库，不含演示数据）。"""
    project = project_by_id(conn, project_id) or {}
    snap = model_config_snapshot()
    df = load_insight_df(conn, project_id, version)
    stats = product_stats(conn, project_id, version)
    counts = status_counts(conn, project_id, version)
    cmp_rows = product_compare_rows(df) if not df.empty else []
    tc = theme_counts(df) if not df.empty else {"counts": {}, "other_only": 0, "other_occ": 0}
    sev = severity_counts(df) if not df.empty else {}
    sent = df["sentiment"].value_counts().to_dict() if not df.empty else {}
    quote_rate = round(float(df["quote_ok"].mean()) * 100, 1) if not df.empty else 0.0
    review_n = int(df["needs_human_review"].sum()) if not df.empty else 0
    prios = priority_rows(conn, project_id, version)
    acts = list_actions(conn, project_id)

    coverage = []
    for s in stats:
        dmin, dmax = str(s["dmin"])[:10], str(s["dmax"])[:10]
        days = 0
        if s["dmin"] and s["dmax"]:
            days = (pd.to_datetime(s["dmax"], utc=True)
                    - pd.to_datetime(s["dmin"], utc=True)).days + 1
        win = f"约 {days} 天" if days else "未知"
        coverage.append({
            "产品": s["name"], "样本量": s["samples"],
            "最早评论": dmin if dmin not in ("None", "nan") else "-",
            "最晚评论": dmax if dmax not in ("None", "nan") else "-",
            "覆盖天数": days,
            "时间窗提示": f"{win}；窗口过短，占比差异不代表长期趋势",
        })

    quality_rows = [
        {"分类": "分析状态", "指标": "样本总量", "数值": counts["samples"]},
        {"分类": "分析状态", "指标": "done", "数值": counts["done"]},
        {"分类": "分析状态", "指标": "degraded（降级）", "数值": counts["degraded"]},
        {"分类": "分析状态", "指标": "skipped（无有效内容）", "数值": counts["skipped"]},
        {"分类": "分析状态", "指标": "failed", "数值": counts["failed"]},
        {"分类": "分析状态", "指标": "待分析", "数值": counts["pending"]},
        {"分类": "人工复核", "指标": "needs_human_review=1", "数值": review_n},
        {"分类": "证据", "指标": "引文命中率（AI 引文可在原文找到）", "数值": f"{quote_rate}%"},
        {"分类": "其他桶", "指标": "仅 other 的评论", "数值": tc["other_only"]},
        {"分类": "其他桶", "指标": "other 被提及次数", "数值": tc["other_occ"]},
        {"分类": "其他桶", "指标": "说明", "数值": "other 为证据不足桶，不参与主题排行"},
    ]
    for k, v in sorted(sev.items()):
        quality_rows.append({"分类": "严重度分布", "指标": SEVERITY_LABELS[k], "数值": v})
    for k, v in sent.items():
        quality_rows.append({"分类": "情感分布", "指标": SENTIMENT_LABELS.get(k, k), "数值": int(v)})

    insight_rows = []
    for code, n in sorted(tc["counts"].items(), key=lambda x: -x[1]):
        part = df[df["themes"].map(lambda t: code in t)]
        avg = part["severity"].dropna()
        insight_rows.append({
            "主题代码": code, "主题": theme_label(code), "提及条数": n,
            "占样本比例": f"{round(n / max(1, len(df)) * 100, 1)}%",
            "平均严重度": round(float(avg.mean()), 2) if len(avg) else None,
            "涉及产品": "、".join(sorted(part["product"].unique())),
        })

    pain_rows = [{
        "痛点（AI 原文短语）": c["label"], "条数": c["count"],
        "平均严重度": c["avg_severity"], "产品": "、".join(c["products"]),
        "feedback_id": "、".join(str(x) for x in c["ids"][:20]),
        "证据强度": "证据 ≥3 条" if c["count"] >= 3 else "弱信号（<3 条，不作结论）",
    } for c in pain_clusters(df)]

    compare_rows = []
    for c in cmp_rows:
        compare_rows.append({
            "产品": c["product"], "样本量": c["samples"],
            "评论时间范围": f"{_fmt_dt(c['dmin'])[:10]} ~ {_fmt_dt(c['dmax'])[:10]}",
            "覆盖天数": c["days"],
            "主要主题 Top5": "；".join(f"{t} {n}" for t, n in c["top_themes"]) or "-",
            "情感分布": "；".join(f"{SENTIMENT_LABELS.get(k, k)} {v}"
                              for k, v in sorted(c["sentiment"].items())) or "-",
            "平均严重度": c["avg_severity"],
            "严重度≥4 占比": f"{round(c['sev4_plus'] / max(1, c['samples']) * 100, 1)}%",
            "仅 other": c["other_only"],
            "备注": "各产品时间窗不同，差异仅描述本窗口，不可外推为长期趋势",
        })

    prio_rows = []
    for i, r in enumerate(prios, 1):
        prio_rows.append({
            "排序": i, "问题": r["title"], "主题": theme_label(r["theme"]),
            "证据条数": r["count"], "AI 分级": r["ai_tier"],
            "最终分级": r["pm_override_tier"] or r["ai_tier"],
            "总分": r["total_score"], "频率分(0-5)": r["freq_score"],
            "严重度(0-5)": r["severity_score"], "核心体验影响(1-5)": r["impact_score"],
            "影响范围(1-5)": r["scope_score"], "成本(1-5)": r["cost_score"],
            "判断理由": r["rationale"] or "（未保存，按默认参数计算）",
            "人工备注": r["pm_note"] or "",
        })

    action_rows = []
    for issue_id, a in acts.items():
        action_rows.append({
            "问题": a["title"], "建议措施": a["proposal"],
            "预期体验改善": a["expected_effect"], "假设": a["assumptions"],
            "不确定性": a["uncertainty"], "优先级": a["priority_tier"] or "-",
            "状态": {"ai_draft": "AI 草案（未确认）", "confirmed": "人工已确认",
                     "rejected": "人工已驳回"}.get(a["status"], a["status"]),
            "确认人": a["decided_by"] or "-", "模型": a["model_name"] or "-",
            "提示词版本": a["prompt_version"] or "-",
        })

    limits = [
        "数据来源为公开应用商店评论，地区、语言与平台分布受限，不能代表全部用户。",
        "各产品评论时间窗不同（ChatGPT 约 4 天、Gemini 约 3 天、Perplexity 约 1 个月），"
        "横向差异只描述本窗口内的样本，不能解读为长期趋势或产品优劣定论。",
        "样本量有限（每产品 150 条），小样本差异不具备统计显著性。",
        "AI 主题、情感、严重度均为模型解释，不是事实；needs_human_review=1 的条目需人工确认。",
        "other 为证据不足桶，不参与主题排行；仅 other 的评论无法形成可执行洞察。",
        "相关性不等于因果；本报告不推断因果机制。",
        "所有结论可回溯到 feedback_id 与原文，未做统计加权或推断性填补。",
    ]

    return {
        "overview": pd.DataFrame([
            {"字段": "项目名称", "内容": project.get("name") or "-"},
            {"字段": "项目描述", "内容": project.get("description") or "-"},
            {"字段": "项目状态", "内容": project.get("status") or "-"},
            {"字段": "分析版本", "内容": version},
            {"字段": "模型", "内容": snap["model"]},
            {"字段": "提示词版本", "内容": snap["prompt_version"]},
            {"字段": "temperature / num_ctx",
             "内容": f"{snap['temperature']} / {snap['num_ctx']}"},
            {"字段": "样本总量", "内容": counts["samples"]},
            {"字段": "产品数量", "内容": len(stats)},
            {"字段": "导出时间(UTC)", "内容": utc_now()},
            {"字段": "口径说明", "内容": "数字全部来自本地 SQLite 真实评论与模型输出；"
                                        "AI 解释与事实分开标识；不编造、不填补。"},
        ]),
        "coverage": pd.DataFrame(coverage),
        "quality": pd.DataFrame(quality_rows),
        "insights": pd.DataFrame(insight_rows),
        "pains": pd.DataFrame(pain_rows),
        "compare": pd.DataFrame(compare_rows),
        "priorities": pd.DataFrame(prio_rows),
        "actions": pd.DataFrame(action_rows),
        "limits": pd.DataFrame([{"序号": i, "说明": t} for i, t in enumerate(limits, 1)]),
    }


SHEET_NAMES = {
    "overview": "项目概述",
    "coverage": "数据来源与样本覆盖",
    "quality": "数据质量",
    "insights": "核心洞察",
    "pains": "痛点",
    "compare": "竞品对比",
    "priorities": "优先级",
    "actions": "改进建议",
    "limits": "局限说明",
}


def build_excel(conn: sqlite3.Connection, project_id: int, version: str) -> dict:
    """生成真实 .xlsx 文件、落盘并写导出日志。"""
    frames = build_report_frames(conn, project_id, version)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"FeedbackLens_p{project_id}_{version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    path = EXPORT_DIR / filename
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for key, name in SHEET_NAMES.items():
            df = frames[key]
            df.to_excel(writer, sheet_name=name, index=False)
            ws = writer.sheets[name]
            for idx, col in enumerate(df.columns, 1):
                width = max([len(str(col))] + [len(str(v)) for v in df[col].head(60)]) + 2
                ws.column_dimensions[
                    ws.cell(row=1, column=idx).column_letter].width = min(60, width)
    rows = int(sum(len(df) for df in frames.values()))
    log_export(conn, project_id, "xlsx", filename, str(path), rows)
    return {"path": str(path), "filename": filename, "rows": rows,
            "bytes": path.read_bytes()}


def build_html(conn: sqlite3.Connection, project_id: int, version: str) -> dict:
    """免费替代方案：自包含可打印 HTML（浏览器另存为 PDF，无需付费库）。"""
    frames = build_report_frames(conn, project_id, version)
    parts = []
    for key, name in SHEET_NAMES.items():
        df = frames[key]
        parts.append(f"<h2>{name}</h2>")
        if df.empty:
            parts.append("<p class='empty'>本部分暂无数据。</p>")
        else:
            parts.append(df.to_html(index=False, border=0, classes="tbl", escape=True))
    html = (
        "<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>"
        f"<title>FeedbackLens 报告 · 项目 {project_id}</title><style>"
        "body{font-family:'Segoe UI','Microsoft YaHei',sans-serif;color:#101828;"
        "margin:32px auto;max-width:1100px;font-size:13px}"
        "h1{font-size:22px}h2{font-size:16px;margin-top:28px;border-left:4px solid #0F5C8C;"
        "padding-left:8px}.tbl{border-collapse:collapse;width:100%;margin-top:8px}"
        ".tbl th,.tbl td{border:1px solid #E3E8EF;padding:6px 8px;text-align:left;"
        "vertical-align:top}.tbl th{background:#F6F8FB}.empty{color:#667085}"
        "@media print{body{margin:0}h2{page-break-after:avoid}}"
        "</style></head><body>"
        f"<h1>FeedbackLens AI 洞察报告</h1><p>版本 {version} · 生成于 {utc_now()}（UTC）"
        " · 数据来自本地 SQLite 真实评论与模型输出</p>" + "".join(parts) +
        "</body></html>"
    )
    filename = f"FeedbackLens_p{project_id}_{version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    path = EXPORT_DIR / filename
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    rows = int(sum(len(df) for df in frames.values()))
    log_export(conn, project_id, "html", filename, str(path), rows)
    return {"path": str(path), "filename": filename, "rows": rows,
            "bytes": html.encode("utf-8")}
