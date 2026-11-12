"""数据读取、质量画像、分层抽样与入库（领域层，禁止依赖任何 UI）。"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import uuid
from pathlib import Path

import pandas as pd

from core import db as dbm

# 参与对比的产品（由项目配置传入，不在此写死数量）
DEFAULT_PRODUCTS = ["ChatGPT_OpenAI", "Google_Gemini", "Perplexity_AI"]

# 分层抽样配额（每个产品内部按星级分配），合计 = 每产品样本量
DEFAULT_ALLOCATION = {1: 40, 2: 20, 3: 25, 4: 25, 5: 40}

MIN_CONTENT_LEN = 10          # 低于此长度视为低信息量，不进入抽样池
SENSITIVE_PATTERNS = [
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),   # 邮箱
    re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"),  # 北美电话
    re.compile(r"https?://\S+"),               # URL
]


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


def read_raw_csv(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            df = pd.read_csv(path, encoding=enc, dtype=str, keep_default_na=False)
            return df
        except UnicodeDecodeError:
            continue
    raise RuntimeError("CSV 编码无法识别（已尝试 utf-8-sig / utf-8 / gb18030）")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def profile(df: pd.DataFrame) -> dict:
    """生成数据质量画像（全部为实际统计，不含推测）。"""
    content = df["content"].astype(str).map(normalize_text)
    ts = pd.to_datetime(df["at_utc"], errors="coerce", utc=True)
    score_num = pd.to_numeric(df["score"], errors="coerce")
    ascii_ratio = float(content.str.replace(r"[^\x00-\x7F]", "", regex=True).str.len().sum() / max(1, content.str.len().sum()))

    per_app = {}
    for app, g in df.groupby("app_name"):
        g_ts = pd.to_datetime(g["at_utc"], errors="coerce", utc=True)
        per_app[str(app)] = {
            "rows": int(len(g)),
            "score_counts": {str(k): int(v) for k, v in pd.to_numeric(g["score"], errors="coerce").value_counts().sort_index().items()},
            "date_min": str(g_ts.min()),
            "date_max": str(g_ts.max()),
            "median_len": float(g["content"].astype(str).str.len().median()),
            "short_lt5": int((g["content"].astype(str).str.len() < 5).sum()),
        }

    week = ts.dt.to_period("W").astype(str).value_counts().sort_index()
    return {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "missing": {c: int((df[c].astype(str).str.strip() == "").sum()) for c in df.columns},
        "app_counts": {str(k): int(v) for k, v in df["app_name"].value_counts().items()},
        "score_counts": {str(k): int(v) for k, v in score_num.value_counts().sort_index().items()},
        "score_out_of_range": int((score_num.isin([1, 2, 3, 4, 5]) == False).sum()),
        "date_parse_fail": int(ts.isna().sum()),
        "date_min": str(ts.min()),
        "date_max": str(ts.max()),
        "week_counts": {str(k): int(v) for k, v in week.items()},
        "duplicate_review_id": int(df["review_id"].duplicated().sum()),
        "duplicate_content": int(content.duplicated().sum()),
        "content_len_median": float(content.str.len().median()),
        "content_len_max": int(content.str.len().max()),
        "content_short_lt5": int((content.str.len() < 5).sum()),
        "content_short_lt10": int((content.str.len() < MIN_CONTENT_LEN).sum()),
        "ascii_char_ratio": round(ascii_ratio, 4),
        "sensitive_hits": int(content.map(lambda t: any(p.search(t) for p in SENSITIVE_PATTERNS)).sum()),
        "per_app": per_app,
    }


def stratified_sample(
    df: pd.DataFrame,
    products: list[str],
    per_product_quota: int,
    allocation: dict[int, int],
    seed: int = 42,
    min_len: int = MIN_CONTENT_LEN,
) -> tuple[pd.DataFrame, dict]:
    """按产品 × 星级分层抽样；配额不足时按实际可用数抽取，不补造数据。"""
    if sum(allocation.values()) != per_product_quota:
        raise ValueError(f"分层配额合计 {sum(allocation.values())} 与每产品样本量 {per_product_quota} 不一致")

    work = df[df["app_name"].isin(products)].copy()
    work["content"] = work["content"].astype(str).map(normalize_text)
    work["_score"] = pd.to_numeric(work["score"], errors="coerce")
    eligible = work[work["content"].str.len() >= min_len].copy()

    parts, shortfall = [], {}
    for app in products:
        g = eligible[eligible["app_name"] == app]
        for star, q in allocation.items():
            layer = g[g["_score"] == star]
            take = min(q, len(layer))
            if take < q:
                shortfall[f"{app}|{star}"] = {"wanted": q, "available": int(len(layer))}
            if take:
                parts.append(layer.sample(n=take, random_state=seed))
    sample = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=work.columns)
    sample = sample.sort_values(["app_name", "_score", "review_id"]).reset_index(drop=True)

    summary = {
        "seed": seed,
        "per_product_quota": per_product_quota,
        "allocation": {str(k): v for k, v in allocation.items()},
        "min_content_len": min_len,
        "eligible_rows": int(len(eligible)),
        "excluded_short_rows": int(len(work) - len(eligible)),
        "sample_rows": int(len(sample)),
        "sample_per_app": {str(k): int(v) for k, v in sample["app_name"].value_counts().items()} if len(sample) else {},
        "sample_per_app_score": {
            app: {str(int(k)): int(v) for k, v in sample[sample["app_name"] == app]["_score"].value_counts().sort_index().items()}
            for app in sample["app_name"].unique()
        } if len(sample) else {},
        "shortfall": shortfall,
    }
    return sample, summary


def _content_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).lower().encode("utf-8")).hexdigest()


def import_sample(
    conn: sqlite3.Connection,
    sample: pd.DataFrame,
    project_name: str,
    source_file: str,
    quality: dict,
    batch_id: str | None = None,
) -> dict:
    """把抽样结果写入 SQLite；隐私字段（user_name / user_image）在入库前丢弃。"""
    batch_id = batch_id or str(uuid.uuid4())
    now = dbm.utc_now()

    cur = conn.execute(
        "INSERT INTO projects (name, description, status, created_at, updated_at) VALUES (?,?,?,?,?)",
        (project_name, "P1 数据集导入", "analyzed", now, now),
    )
    project_id = cur.lastrowid

    product_ids = {}
    for app in sorted(sample["app_name"].unique()):
        pkg = sample[sample["app_name"] == app]["package_name"].iloc[0]
        cur = conn.execute(
            "INSERT INTO products (project_id, name, package_name) VALUES (?,?,?)",
            (project_id, app, pkg),
        )
        product_ids[app] = cur.lastrowid

    cur = conn.execute(
        "INSERT INTO import_batches (id, project_id, filename, row_count, accepted_count, rejected_count, quality_report_json, created_at) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (batch_id, project_id, source_file, int(len(sample)), int(len(sample)), 0, json.dumps(quality, ensure_ascii=False), now),
    )

    rows = []
    for r in sample.itertuples(index=False):
        meta = {
            "thumbs_up_count": str(getattr(r, "thumbs_up_count", "")),
            "review_created_version": str(getattr(r, "review_created_version", "")),
            "app_version": str(getattr(r, "app_version", "")),
            "scrape_lang": str(getattr(r, "scrape_lang", "")),
            "scrape_country": str(getattr(r, "scrape_country", "")),
            "scraped_at_utc": str(getattr(r, "scraped_at_utc", "")),
        }
        review_at = pd.to_datetime(getattr(r, "at_utc"), errors="coerce", utc=True)
        rows.append((
            project_id, batch_id, str(r.review_id), product_ids[str(r.app_name)],
            normalize_text(r.content), int(float(r.score)) if str(r.score).strip() else None,
            review_at.isoformat() if pd.notna(review_at) else None,
            "en", "mendeley", json.dumps(meta, ensure_ascii=False),
            _content_hash(r.content), now,
        ))

    conn.executemany(
        "INSERT OR IGNORE INTO feedback (project_id, batch_id, review_id, product_id, content, score, review_at, lang, source, meta_json, content_hash, created_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()
    return {
        "project_id": project_id,
        "batch_id": batch_id,
        "inserted": conn.execute("SELECT COUNT(*) FROM feedback WHERE project_id=?", (project_id,)).fetchone()[0],
    }
