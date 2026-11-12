"""SQLite 建库与基础操作（领域层，禁止依赖任何 UI）。"""
from __future__ import annotations

import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS projects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT,
    status TEXT NOT NULL DEFAULT 'draft',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    deleted_at TEXT
);

CREATE TABLE IF NOT EXISTS products (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL REFERENCES projects(id),
    name TEXT NOT NULL,
    package_name TEXT,
    UNIQUE(project_id, name)
);

CREATE TABLE IF NOT EXISTS import_batches (
    id TEXT PRIMARY KEY,
    project_id INTEGER NOT NULL REFERENCES projects(id),
    filename TEXT,
    row_count INTEGER,
    accepted_count INTEGER,
    rejected_count INTEGER,
    quality_report_json TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL REFERENCES projects(id),
    batch_id TEXT REFERENCES import_batches(id),
    review_id TEXT NOT NULL,
    product_id INTEGER NOT NULL REFERENCES products(id),
    content TEXT NOT NULL,
    score INTEGER,
    review_at TEXT,
    lang TEXT,
    source TEXT,
    meta_json TEXT,
    content_hash TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(project_id, review_id)
);

CREATE TABLE IF NOT EXISTS analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    feedback_id INTEGER NOT NULL REFERENCES feedback(id),
    run_id TEXT,
    model_name TEXT,
    model_version TEXT,
    prompt_version TEXT,
    themes_json TEXT,
    sentiment TEXT,
    intent TEXT,
    pain_point TEXT,
    feature_request TEXT,
    severity INTEGER,
    evidence_quote TEXT,
    confidence REAL,
    needs_human_review INTEGER NOT NULL DEFAULT 0,
    is_valid INTEGER NOT NULL DEFAULT 0,
    raw_output TEXT,
    error TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL,
    UNIQUE(feedback_id, model_name, prompt_version)
);

CREATE TABLE IF NOT EXISTS issues (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL REFERENCES projects(id),
    title TEXT NOT NULL,
    theme TEXT,
    evidence_ids_json TEXT,
    evidence_count INTEGER,
    products_json TEXT,
    status TEXT NOT NULL DEFAULT 'candidate',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS priorities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL REFERENCES projects(id),
    issue_id INTEGER NOT NULL REFERENCES issues(id),
    freq_score REAL,
    severity_score REAL,
    impact_score REAL,
    scope_score REAL,
    cost_score REAL,
    total_score REAL,
    ai_tier TEXT,
    pm_override_tier TEXT,
    rationale TEXT,
    pm_note TEXT,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS actions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL REFERENCES projects(id),
    issue_id INTEGER NOT NULL REFERENCES issues(id),
    proposal TEXT,
    expected_effect TEXT,
    assumptions TEXT,
    uncertainty TEXT,
    status TEXT NOT NULL DEFAULT 'ai_draft',
    decided_by TEXT,
    decided_at TEXT,
    reject_reason TEXT
);

CREATE TABLE IF NOT EXISTS eval_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    sample_size INTEGER NOT NULL,
    labeler TEXT,
    metrics_json TEXT,
    confusion_json TEXT,
    model_name TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS eval_labels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    feedback_id INTEGER NOT NULL REFERENCES feedback(id),
    labeler TEXT,
    gold_themes_json TEXT,
    gold_sentiment TEXT,
    gold_intent TEXT,
    gold_severity INTEGER,
    notes TEXT,
    labeled_at TEXT,
    UNIQUE(feedback_id, labeler)
);

CREATE INDEX IF NOT EXISTS idx_feedback_project ON feedback(project_id);
CREATE INDEX IF NOT EXISTS idx_feedback_product ON feedback(product_id);
CREATE INDEX IF NOT EXISTS idx_analyses_feedback ON analyses(feedback_id);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    """轻量迁移：已存在的库补齐新增列（P2 断点续跑需要 attempts）。"""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(analyses)")}
    if "attempts" not in cols:
        conn.execute("ALTER TABLE analyses ADD COLUMN attempts INTEGER NOT NULL DEFAULT 0")
    conn.commit()


def init_db(db_path: Path) -> sqlite3.Connection:
    conn = connect(db_path)
    conn.executescript(SCHEMA_SQL)
    _migrate(conn)
    conn.commit()
    return conn


def backup_if_exists(db_path: Path) -> Path | None:
    """写入前备份：存在则复制为 .bak（保留最近一次）。

    Windows 下 .bak 可能被杀软/其他进程短暂占用，失败时改用带时间戳的备份名，
    不跳过备份。
    """
    if not db_path.exists():
        return None
    bak = db_path.with_suffix(db_path.suffix + ".bak")
    if bak.exists():
        try:
            bak.unlink()
        except OSError:
            pass
    try:
        shutil.copy2(db_path, bak)
        return bak
    except PermissionError:
        bak = db_path.with_suffix(
            db_path.suffix + datetime.now().strftime(".%Y%m%d%H%M%S.bak")
        )
        shutil.copy2(db_path, bak)
        return bak
