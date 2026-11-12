"""P3 质量核查：读 450 条全量分析结果，生成统计与抽样对照报告。

只读数据库，不调用模型、不修改分析结果。
产出：reports/p2_full_450_quality.md
"""
from __future__ import annotations

import json
import re
import sqlite3
import statistics
import sys
from collections import Counter
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "feedbacklens.db"
OUT = ROOT / "reports" / "p2_full_450_quality.md"
MODEL = "qwen3:8b"
PROMPT = "v2.0-p2"
THEMES = [
    "account_login", "pricing_subscription", "answer_quality", "speed_latency",
    "stability_crash", "missing_features", "ui_usability", "privacy_security",
    "customer_support", "sync_platform", "model_limits", "other",
]
SENTIMENTS = ["positive", "neutral", "mixed", "negative"]
INTENTS = ["praise", "complaint", "feature_request", "question", "other"]
STATUS = ["done", "degraded", "failed", "skipped"]


def norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def norm_phrase(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def clip(s, n: int) -> str:
    s = (s or "").replace("\n", " ").replace("|", "／").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def load_rows(conn: sqlite3.Connection) -> list[dict]:
    cur = conn.execute(
        """
        SELECT a.id, a.feedback_id, a.status, a.is_valid, a.needs_human_review,
               a.themes_json, a.sentiment, a.intent, a.pain_point, a.feature_request,
               a.severity, a.evidence_quote, a.confidence, a.attempts, a.error,
               f.content, f.score, f.review_at, f.review_id, p.name AS app
        FROM analyses a
        JOIN feedback f ON f.id = a.feedback_id
        JOIN products p ON p.id = f.product_id
        WHERE a.model_name = ? AND a.prompt_version = ?
        ORDER BY p.name, f.score, f.id
        """,
        (MODEL, PROMPT),
    )
    rows = []
    for r in cur.fetchall():
        d = dict(r)
        try:
            d["themes"] = json.loads(d.get("themes_json") or "[]")
        except json.JSONDecodeError:
            d["themes"] = []
        rows.append(d)
    return rows


def pct(n: int, d: int) -> str:
    return f"{n / d * 100:.1f}%" if d else "n/a"


def repetition(rows: list[dict], field: str) -> dict:
    values = [norm_phrase(r[field]) for r in rows if (r.get(field) or "").strip()]
    total = len(values)
    counter = Counter(values)
    distinct = len(counter)
    dup_share = (total - distinct) / total if total else 0.0
    top = counter.most_common(8)
    # 近义复读：仅对出现 >=2 次的短语做两两相似度（保守阈值 0.9）
    frequent = [v for v, c in counter.items() if c >= 2]
    near_groups = []
    used = set()
    for i, a in enumerate(frequent):
        if a in used:
            continue
        group = [a]
        for b in frequent[i + 1:]:
            if b in used:
                continue
            if SequenceMatcher(None, a, b).ratio() >= 0.9:
                group.append(b)
                used.add(b)
        if len(group) >= 2:
            used.add(a)
            near_groups.append(group)
    return {
        "total": total,
        "distinct": distinct,
        "dup_share": dup_share,
        "top": top,
        "near_groups": near_groups[:5],
    }


def main() -> int:
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    rows = load_rows(conn)
    conn.close()

    if not rows:
        print(json.dumps({"error": "no_rows", "model": MODEL, "prompt": PROMPT}, ensure_ascii=False))
        return 2

    analyzed = [r for r in rows if r["status"] in {"done", "degraded"}]
    status_counts = Counter(r["status"] for r in rows)
    apps = sorted({r["app"] for r in rows})

    # 证据复核（不依赖库里的 is_valid，独立重算）
    quote_mismatch = 0
    for r in analyzed:
        q = norm_ws(r.get("evidence_quote") or "")
        if not q or q not in norm_ws(r.get("content") or ""):
            quote_mismatch += 1

    # 星级 × 情绪 / 严重度
    star_stats = {}
    for s in range(1, 6):
        bucket = [r for r in analyzed if int(r["score"] or 0) == s]
        if not bucket:
            star_stats[s] = {"n": 0}
            continue
        sent = Counter(r["sentiment"] for r in bucket)
        sev = [r["severity"] for r in bucket if r["severity"] is not None]
        star_stats[s] = {
            "n": len(bucket),
            "sent": sent,
            "neg_share": sent.get("negative", 0) / len(bucket),
            "pos_share": sent.get("positive", 0) / len(bucket),
            "sev_mean": round(statistics.mean(sev), 2) if sev else None,
            "sev_zero": sum(1 for x in sev if x == 0),
            "sev_five": sum(1 for x in sev if x == 5),
            "sev_none": len(bucket) - len(sev),
        }

    # 产品维度
    app_stats = {}
    for app in apps:
        bucket = [r for r in rows if r["app"] == app]
        ab = [r for r in bucket if r["status"] in {"done", "degraded"}]
        dates = sorted({str(r["review_at"])[:10] for r in bucket if r["review_at"]})
        themes = Counter(t for r in ab for t in r["themes"])
        sent = Counter(r["sentiment"] for r in ab)
        intent = Counter(r["intent"] for r in ab)
        sev = [r["severity"] for r in ab if r["severity"] is not None]
        app_stats[app] = {
            "n": len(bucket),
            "analyzed": len(ab),
            "status": Counter(r["status"] for r in bucket),
            "first": dates[0] if dates else None,
            "last": dates[-1] if dates else None,
            "days": len(dates),
            "stars": Counter(int(r["score"] or 0) for r in bucket),
            "sent": sent,
            "intent": intent,
            "themes": themes,
            "sev": Counter(sev),
            "sev_mean": round(statistics.mean(sev), 2) if sev else None,
            "low_conf": sum(1 for r in ab if (r["confidence"] or 0) < 0.5),
            "nhr": sum(1 for r in ab if r["needs_human_review"]),
            "flags": sum(1 for r in ab if (r["error"] or "").startswith(("flag:", "star_"))) + sum(
                1 for r in ab if "flag:" in (r["error"] or "")
            ),
        }

    # 严重度总体
    sev_all = [r["severity"] for r in analyzed if r["severity"] is not None]
    sev_counter = Counter(sev_all)

    # 复读
    pain_rep = repetition(analyzed, "pain_point")
    feat_rep = repetition(analyzed, "feature_request")

    # 主题总体
    theme_counter = Counter(t for r in analyzed for t in r["themes"])

    # 抽 10 条人工对照：每个星级 2 条，尽量覆盖不同产品
    samples = []
    for s in range(1, 6):
        used_apps = set()
        picked = 0
        for r in analyzed:
            if int(r["score"] or 0) != s or r["app"] in used_apps:
                continue
            samples.append(r)
            used_apps.add(r["app"])
            picked += 1
            if picked == 2:
                break
    samples = samples[:10]

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# P2 全量 450 条质量核查报告",
        "",
        f"- 生成时间：{now}",
        f"- 模型 / prompt：`{MODEL}` / `{PROMPT}`",
        f"- 分析行数：{len(rows)}（去重后按 `feedback_id + model + prompt` 唯一）",
        f"- 状态：done {status_counts.get('done', 0)} / degraded {status_counts.get('degraded', 0)} / "
        f"failed {status_counts.get('failed', 0)} / skipped {status_counts.get('skipped', 0)}",
        f"- 证据独立复核：{len(analyzed) - quote_mismatch}/{len(analyzed)} 条引文仍是原文连续子串",
        "",
        "## 1. 各产品样本量与时间窗口",
        "",
        "| 产品 | 分析条数 | 时间窗口 | 覆盖天数 | done/degraded/failed/skipped |",
        "|---|---|---|---|---|",
    ]
    for app in apps:
        st = app_stats[app]
        lines.append(
            f"| {app} | {st['n']}（分析 {st['analyzed']}） | {st['first']} ~ {st['last']} | {st['days']} | "
            f"{st['status'].get('done', 0)}/{st['status'].get('degraded', 0)}/{st['status'].get('failed', 0)}/{st['status'].get('skipped', 0)} |"
        )
    lines += [
        "",
        "> **时间窗口公平性**：三产品样本量均为 150，但时间覆盖差异显著。"
        f"ChatGPT 覆盖 {app_stats.get('ChatGPT_OpenAI', {}).get('days')} 天、"
        f"Gemini 覆盖 {app_stats.get('Google_Gemini', {}).get('days')} 天、"
        f"Perplexity 覆盖 {app_stats.get('Perplexity_AI', {}).get('days')} 天。"
        "因此任何“时间趋势”都不具备统计意义，汇总与报告只能标注窗口与覆盖，不能做趋势推断。",
        "",
        "## 2. 星级 × 情绪 / 严重度一致性",
        "",
        "| 星级 | 分析条数 | 负面占比 | 正面占比 | 平均严重度 | severity=0 | severity=5 | severity 缺失 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in range(1, 6):
        st = star_stats[s]
        if st["n"] == 0:
            lines.append(f"| {s}★ | 0 | - | - | - | - | - | - |")
            continue
        lines.append(
            f"| {s}★ | {st['n']} | {pct(st['sent'].get('negative', 0), st['n'])} | "
            f"{pct(st['sent'].get('positive', 0), st['n'])} | {st['sev_mean']} | "
            f"{st['sev_zero']} | {st['sev_five']} | {st['sev_none']} |"
        )

    # 自动判定
    low = [r for r in analyzed if int(r["score"] or 0) <= 2]
    high = [r for r in analyzed if int(r["score"] or 0) >= 4]
    neg_low = sum(1 for r in low if r["sentiment"] == "negative") / len(low) if low else 0
    pos_high = sum(1 for r in high if r["sentiment"] == "positive") / len(high) if high else 0
    zero_share = sev_counter.get(0, 0) / len(sev_all) if sev_all else 0
    five_share = sev_counter.get(5, 0) / len(sev_all) if sev_all else 0

    lines += [
        "",
        f"- 1–2★ 中负面占比：**{neg_low * 100:.1f}%**（{len(low)} 条）",
        f"- 4–5★ 中正面占比：**{pos_high * 100:.1f}%**（{len(high)} 条）",
        f"- severity=0 占比：**{zero_share * 100:.1f}%**；severity=5 占比：**{five_share * 100:.1f}%**；"
        f"平均严重度：**{round(statistics.mean(sev_all), 2) if sev_all else 'n/a'}**",
        "",
        "## 3. 主题 / 情绪 / 意图分布",
        "",
        "### 3.1 主题总体分布（多标签）",
        "",
        "| 主题 | 次数 | 占分析条数 |",
        "|---|---|---|",
    ]
    for t in THEMES:
        n = theme_counter.get(t, 0)
        lines.append(f"| `{t}` | {n} | {pct(n, len(analyzed))} |")

    lines += [
        "",
        "### 3.2 各产品 Top 5 主题",
        "",
        "| 产品 | Top 主题（次数） |",
        "|---|---|",
    ]
    for app in apps:
        top = app_stats[app]["themes"].most_common(5)
        lines.append(f"| {app} | " + "；".join(f"{t}({n})" for t, n in top) + " |")

    lines += [
        "",
        "### 3.3 情绪 / 意图分布",
        "",
        "| 产品 | positive | neutral | mixed | negative | praise | complaint | feature_request | question | other |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for app in apps:
        st = app_stats[app]
        s = st["sent"]
        i = st["intent"]
        lines.append(
            f"| {app} | {s.get('positive', 0)} | {s.get('neutral', 0)} | {s.get('mixed', 0)} | {s.get('negative', 0)} | "
            f"{i.get('praise', 0)} | {i.get('complaint', 0)} | {i.get('feature_request', 0)} | {i.get('question', 0)} | {i.get('other', 0)} |"
        )

    lines += [
        "",
        "## 4. 严重程度分布",
        "",
        "| severity | 条数 | 占比 |",
        "|---|---|---|",
    ]
    for s in range(6):
        lines.append(f"| {s} | {sev_counter.get(s, 0)} | {pct(sev_counter.get(s, 0), len(sev_all))} |")
    lines += [
        "",
        f"各产品平均严重度：" + "；".join(
            f"{app} {app_stats[app]['sev_mean']}" for app in apps
        ),
        "",
        "## 5. 复读检测（跨评论重复）",
        "",
        "### 5.1 pain_point",
        "",
        f"- 非空 {pain_rep['total']} 条，去重后 {pain_rep['distinct']} 条，**重复占比 {pain_rep['dup_share'] * 100:.1f}%**",
        "",
        "| 高频文本 | 次数 |",
        "|---|---|",
    ]
    for text, n in pain_rep["top"]:
        lines.append(f"| {clip(text, 80)} | {n} |")
    lines += [
        "",
        "近义重复组（相似度 ≥0.9）：" + (
            "无" if not pain_rep["near_groups"] else
            "；".join(" / ".join(clip(x, 50) for x in g) for g in pain_rep["near_groups"])
        ),
        "",
        "### 5.2 feature_request",
        "",
        f"- 非空 {feat_rep['total']} 条，去重后 {feat_rep['distinct']} 条，**重复占比 {feat_rep['dup_share'] * 100:.1f}%**",
        "",
        "| 高频文本 | 次数 |",
        "|---|---|",
    ]
    for text, n in feat_rep["top"]:
        lines.append(f"| {clip(text, 80)} | {n} |")
    lines += [
        "",
        "近义重复组（相似度 ≥0.9）：" + (
            "无" if not feat_rep["near_groups"] else
            "；".join(" / ".join(clip(x, 50) for x in g) for g in feat_rep["near_groups"])
        ),
        "",
        "## 6. 抽 10 条：原文与 AI 输出对照",
        "",
        "抽取规则：每星级 2 条，尽量覆盖不同产品。引文命中为自动校验；语义一致性见下方人工判断。",
        "",
    ]
    for i, r in enumerate(samples, 1):
        lines += [
            f"### 6.{i} feedback_id={r['feedback_id']} / {r['app']} / {r['score']}★",
            "",
            f"- 原文：{clip(r['content'], 260)}",
            f"- 引文：{clip(r['evidence_quote'], 160)}",
            f"- 主题：{', '.join(r['themes'])}；情感：{r['sentiment']}；意图：{r['intent']}；严重度：{r['severity']}；置信度：{r['confidence']}",
            f"- pain_point：{clip(r['pain_point'], 120) or '（空）'}；feature_request：{clip(r['feature_request'], 120) or '（空）'}",
            f"- 自动校验：引文{'命中' if norm_ws(r['evidence_quote'] or '') in norm_ws(r['content'] or '') else '未命中'}；"
            f"状态 `{r['status']}`；needs_human_review={r['needs_human_review']}",
        ]
        lines.append("")

    lines += [
        "## 7. 人工判断（待填）",
        "",
        "<!-- HUMAN_REVIEW -->",
        "",
        "## 8. 说明与局限",
        "",
        "- 本报告只覆盖本批 450 条样本，样本为美国区英文评论，**不能外推到中国用户或其他平台**。",
        "- 时间窗口：ChatGPT 4 天、Gemini 3 天，任何趋势推断都无效。",
        "- 复读检测基于规范化后的精确重复 + 0.9 相似度近义组；同义改写但字面不同的复读可能漏检。",
        "- 人工对照仅 10 条，用于发现系统性偏差，不构成准确率估计（不是 60 条标注评估）。",
    ]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({
        "rows": len(rows),
        "status": dict(status_counts),
        "quote_mismatch": quote_mismatch,
        "neg_share_1_2": round(neg_low, 3),
        "pos_share_4_5": round(pos_high, 3),
        "severity_zero_share": round(zero_share, 3),
        "severity_five_share": round(five_share, 3),
        "pain_point_dup_share": round(pain_rep["dup_share"], 3),
        "feature_request_dup_share": round(feat_rep["dup_share"], 3),
        "top_theme": theme_counter.most_common(3),
        "report": str(OUT),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
