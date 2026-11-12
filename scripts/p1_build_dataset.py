"""P1：读取原始 CSV → 质量画像 → 分层抽样 → 建库 → 产出报告。

用法（项目根目录）：python scripts/p1_build_dataset.py
所有参数来自下方 CONFIG，样本量为配置项（可扩到 300/款），不写死。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from core import db as dbm
from core import ingest

CONFIG = {
    "raw_csv": ROOT / "data" / "raw" / "reviews_all_GenAI.csv",
    "db_path": ROOT / "data" / "feedbacklens.db",
    "sample_csv": ROOT / "data" / "samples" / "sample_450.csv",
    "report_md": ROOT / "reports" / "data_quality_p1.md",
    "products": ["ChatGPT_OpenAI", "Google_Gemini", "Perplexity_AI"],
    "per_product_quota": 150,          # 可改为 300 扩量
    "allocation": {1: 40, 2: 20, 3: 25, 4: 25, 5: 40},
    "seed": 42,
    "project_name": "GenAI 三产品评论洞察（P1）",
}


def md_table(rows: list[tuple], headers: list[str]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def main() -> None:
    cfg = CONFIG
    csv_path: Path = cfg["raw_csv"]
    if not csv_path.exists():
        raise SystemExit(f"原始 CSV 不存在：{csv_path}")

    df = ingest.read_raw_csv(csv_path)
    quality = ingest.profile(df)
    quality["file"] = csv_path.name
    quality["file_size_mb"] = round(csv_path.stat().st_size / 1024 / 1024, 2)
    quality["file_sha256"] = ingest.file_sha256(csv_path)

    sample, sample_info = ingest.stratified_sample(
        df,
        products=cfg["products"],
        per_product_quota=cfg["per_product_quota"],
        allocation=cfg["allocation"],
        seed=cfg["seed"],
    )
    if sample.empty:
        raise SystemExit("抽样结果为空，检查 products 配置与数据")

    cfg["sample_csv"].parent.mkdir(parents=True, exist_ok=True)
    keep_cols = ["review_id", "app_name", "package_name", "score", "at_utc", "content", "thumbs_up_count",
                 "review_created_version", "app_version", "scrape_lang", "scrape_country", "scraped_at_utc"]
    sample[keep_cols].to_csv(cfg["sample_csv"], index=False, encoding="utf-8-sig")

    dbm.backup_if_exists(cfg["db_path"])
    conn = dbm.init_db(cfg["db_path"])
    info = ingest.import_sample(conn, sample, cfg["project_name"], csv_path.name, quality)
    db_counts = {
        "feedback": conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0],
        "products": conn.execute("SELECT COUNT(*) FROM products").fetchone()[0],
        "projects": conn.execute("SELECT COUNT(*) FROM projects").fetchone()[0],
    }
    per_product_db = {
        r["name"]: conn.execute(
            "SELECT COUNT(*) FROM feedback f JOIN products p ON p.id=f.product_id WHERE p.name=?", (r["name"],)
        ).fetchone()[0]
        for r in conn.execute("SELECT name FROM products ORDER BY name")
    }
    conn.close()

    # ---------- 生成中文质量报告 ----------
    lines = []
    lines.append("# P1 数据质量报告（实测）\n")
    lines.append(f"- 原始文件：`{quality['file']}`，{quality['file_size_mb']} MB")
    lines.append(f"- SHA256：`{quality['file_sha256']}`")
    lines.append(f"- 实际行数：**{quality['rows']:,}**；字段数：{len(quality['columns'])}")
    lines.append(f"- 日期范围（UTC）：`{quality['date_min']}` ~ `{quality['date_max']}`；解析失败 {quality['date_parse_fail']} 条")
    lines.append(f"- 重复 `review_id`：{quality['duplicate_review_id']} 条；重复正文：{quality['duplicate_content']} 条")
    lines.append(f"- 正文长度中位数：{quality['content_len_median']} 字符；<5 字符 {quality['content_short_lt5']:,} 条；<{ingest.MIN_CONTENT_LEN} 字符 {quality['content_short_lt10']:,} 条")
    lines.append(f"- 英文字符占比（ASCII 比例，粗略）：{quality['ascii_char_ratio']*100:.2f}%")
    lines.append(f"- 疑似含邮箱/电话/URL 的正文：{quality['sensitive_hits']} 条（引用时将回避）\n")

    lines.append("## 1. 各应用条数与评分分布\n")
    rows = []
    for app, v in quality["per_app"].items():
        sc = v["score_counts"]
        rows.append((app, v["rows"], sc.get("1", 0), sc.get("2", 0), sc.get("3", 0), sc.get("4", 0), sc.get("5", 0),
                     v["date_min"][:10], v["date_max"][:10], f"{v['median_len']:.0f}"))
    lines.append(md_table(rows, ["应用", "条数", "1★", "2★", "3★", "4★", "5★", "最早", "最晚", "长度中位"]))
    lines.append("")

    lines.append("## 2. 关键发现（影响后续模块设计）\n")
    lines.append(f"1. **评分严重偏正**：全量中 5★ 占 {quality['score_counts'].get('5', 0)/quality['rows']*100:.1f}%，1★ 占 {quality['score_counts'].get('1', 0)/quality['rows']*100:.1f}%。")
    lines.append("   若纯随机抽 150 条，约 110 条是 5★，评论多为“good/nice”，洞察价值低。因此采用**分层抽样**（1★:40 / 2★:20 / 3★:25 / 4★:25 / 5★:40），属**有意加权**，报告中必须披露“样本不等同总体分布”。")
    chatgpt = quality["per_app"].get("ChatGPT_OpenAI")
    gemini = quality["per_app"].get("Google_Gemini")
    pplx = quality["per_app"].get("Perplexity_AI")
    if chatgpt and gemini and pplx:
        lines.append(f"2. **时间窗口极短且不一致**：ChatGPT `{chatgpt['date_min']}` ~ `{chatgpt['date_max']}`（约 4 天）；"
                     f"Gemini `{gemini['date_min']}` ~ `{gemini['date_max']}`（约 3 天）；"
                     f"Perplexity `{pplx['date_min']}` ~ `{pplx['date_max']}`（约 1 个月）。")
        lines.append("   → **时间趋势模块对 ChatGPT/Gemini 基本不可用**，只能做“同一窗口内的横截面对比”；M4 的“时间变化”需降级为可选，并标注“窗口过短，不足以判断趋势”。")
    lines.append(f"3. **重复正文比例高**：{quality['duplicate_content']:,} / {quality['rows']:,} = {quality['duplicate_content']/quality['rows']*100:.1f}%。短评（如 “good”“nice”）大量重复，属正常现象但**不可当作独立证据加权**。")
    lines.append(f"4. **低信息量评论多**：正文 <5 字符 {quality['content_short_lt5']:,} 条（{quality['content_short_lt5']/quality['rows']*100:.1f}%）。抽样时已排除 <{ingest.MIN_CONTENT_LEN} 字符的条目，该排除属**筛选偏差**，需披露。")
    lines.append(f"5. 开发者回复几乎缺失：`reply_content` 缺失 {quality['missing'].get('reply_content', 0):,} 条（{quality['missing'].get('reply_content', 0)/quality['rows']*100:.1f}%），不能作为分析输入。")
    lines.append("")

    lines.append("## 3. 缺失值（前 10 列）\n")
    miss_rows = sorted(quality["missing"].items(), key=lambda kv: -kv[1])[:10]
    lines.append(md_table([(k, v, f"{v/quality['rows']*100:.2f}%") for k, v in miss_rows], ["字段", "缺失条数", "占比"]))
    lines.append("")

    lines.append("## 4. 抽样方案与结果\n")
    lines.append(f"- 产品：{', '.join(cfg['products'])}；每产品配额 **{cfg['per_product_quota']}**（配置项，可扩到 300）")
    lines.append(f"- 分层配额（每产品）：{cfg['allocation']}；随机种子：{cfg['seed']}")
    lines.append(f"- 可选池：{sample_info['eligible_rows']:,} 条（已排除正文 <{ingest.MIN_CONTENT_LEN} 字符的 {sample_info['excluded_short_rows']:,} 条）")
    lines.append(f"- 实际抽样：**{sample_info['sample_rows']}** 条；每产品：{sample_info['sample_per_app']}")
    if sample_info["shortfall"]:
        lines.append(f"- 配额不足（按实际可用抽取，未补造）：{sample_info['shortfall']}")
    else:
        lines.append("- 配额充足，全部按计划抽满。")
    lines.append("")

    lines.append("## 5. 入库结果\n")
    lines.append(f"- 数据库：`{cfg['db_path']}`")
    lines.append(f"- 表计数：{db_counts}")
    lines.append(f"- 每产品入库条数：{per_product_db}")
    lines.append("- 隐私处理：`user_name`、`user_image` **未入库**（导入前列即丢弃）；仅保留 `review_id` 作追溯。")
    lines.append("")

    lines.append("## 6. 局限（必须随报告披露）\n")
    lines.append("- 地域：美国区（`scrape_country='us'`）；语言：英文（`scrape_lang='en'`）→ **结论不外推到中国用户**。")
    lines.append("- 平台：仅 Google Play；不含 iOS / Web。")
    lines.append("- 规模：全量 70,000 条，本项目仅抽样 450 条（约 0.64%），属**描述性洞察**，不做统计显著性结论。")
    lines.append("- 时间：抓取集中在 2025-10 至 2025-11（ChatGPT/Gemini 仅数天），不支持长期趋势判断。")
    lines.append("- 抽样：按星级分层加权 + 排除超短文本，**不等同总体分布**。")

    cfg["report_md"].parent.mkdir(parents=True, exist_ok=True)
    cfg["report_md"].write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({
        "rows": quality["rows"],
        "columns": len(quality["columns"]),
        "apps": quality["app_counts"],
        "date_range": [quality["date_min"], quality["date_max"]],
        "score_counts": quality["score_counts"],
        "duplicate_content": quality["duplicate_content"],
        "short_lt5": quality["content_short_lt5"],
        "ascii_ratio": quality["ascii_char_ratio"],
        "sample": sample_info,
        "db": {"path": str(cfg["db_path"]), **db_counts, "per_product": per_product_db},
        "report": str(cfg["report_md"]),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
