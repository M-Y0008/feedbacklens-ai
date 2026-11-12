"""v2.2 全量 450 条同口径报告（只读，不调用模型）。

用法：python scripts/p2_v22_full450_report.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.analysis import normalize_ws  # noqa: E402
from core.db import init_db  # noqa: E402

MODEL = "qwen3:8b"
OLD, NEW = "v2.1-p2", "v2.2-p2"
REPORT = ROOT / "reports" / "p2_v22_full450.md"

# 30 条抽读判断：id -> (v2.2 判断, 说明)；人工阅读后填入
JUDGE30: dict[int, tuple[str, str]] = {}


def load(conn, version):
    rows = conn.execute(
        "SELECT a.*, f.content, f.score, p.name AS app "
        "FROM analyses a JOIN feedback f ON f.id=a.feedback_id "
        "JOIN products p ON p.id=f.product_id "
        "WHERE a.model_name=? AND a.prompt_version=?",
        (MODEL, version),
    ).fetchall()
    return [dict(r) for r in rows]


def themes_of(row):
    try:
        t = json.loads(row["themes_json"] or "[]")
    except json.JSONDecodeError:
        return []
    return t if isinstance(t, list) else []


def quote_ok(row):
    q = normalize_ws(row.get("evidence_quote") or "")
    return bool(q) and len(q) >= 4 and q in normalize_ws(row.get("content") or "")


def consistency(rows):
    low = [r for r in rows if r["score"] in (1, 2)]
    high = [r for r in rows if r["score"] in (4, 5)]
    neg = sum(1 for r in low if r["sentiment"] == "negative")
    pos = sum(1 for r in high if r["sentiment"] == "positive")
    conflict = sum(
        1 for r in rows
        if (r["score"] <= 2 and r["sentiment"] == "positive")
        or (r["score"] >= 4 and r["sentiment"] == "negative")
    )
    return {
        "low_n": len(low), "low_neg": neg,
        "low_pct": neg / len(low) * 100 if low else 0,
        "high_n": len(high), "high_pos": pos,
        "high_pct": pos / len(high) * 100 if high else 0,
        "conflict": conflict,
    }


def summarize(rows):
    analyzed = [r for r in rows if r["status"] != "skipped"]
    themes = Counter()
    other_only = 0
    for r in rows:
        t = themes_of(r)
        if t == ["other"]:
            other_only += 1
        themes.update(t)
    sev = Counter(("null" if r["severity"] is None else r["severity"]) for r in analyzed)
    hits = sum(1 for r in analyzed if quote_ok(r))
    return {
        "n": len(rows),
        "status": dict(Counter(r["status"] for r in rows)),
        "other_only": other_only,
        "other_only_pct": other_only / len(rows) * 100 if rows else 0,
        "other_occ": themes.get("other", 0),
        "theme_total": sum(themes.values()),
        "themes": themes,
        "aq": themes.get("answer_quality", 0),
        "ml": themes.get("model_limits", 0),
        "mf": themes.get("missing_features", 0),
        "hits": hits,
        "hit_n": len(analyzed),
        "hit_pct": hits / len(analyzed) * 100 if analyzed else 0,
        "severity": sev,
        "consistency": consistency(analyzed),
    }


def fmt_status(st):
    return f"{st.get('done', 0)} / {st.get('degraded', 0)} / {st.get('failed', 0)}"


def main() -> int:
    conn = init_db(ROOT / "data" / "feedbacklens.db")
    old, new = load(conn, OLD), load(conn, NEW)
    if len(old) != 450 or len(new) != 450:
        print(json.dumps({"error": "not_450", "old": len(old), "new": len(new)}))
        return 2
    so, sn = summarize(old), summarize(new)

    runs = conn.execute(
        "SELECT run_id, COUNT(*) n, MIN(created_at) a, MAX(created_at) b "
        "FROM analyses WHERE model_name=? AND prompt_version=? GROUP BY run_id ORDER BY a",
        (MODEL, NEW),
    ).fetchall()
    main_run = max(runs, key=lambda r: r["n"])
    span = (datetime.fromisoformat(main_run["b"]) - datetime.fromisoformat(main_run["a"])).total_seconds()

    ranked = [(t, n) for t, n in sn["themes"].most_common() if t != "other"]

    lines = [
        "# v2.2-p2 全量 450 条同口径报告",
        "",
        f"- 模型：`{MODEL}`；对比：`{OLD}` → `{NEW}`；样本：450 条",
        f"- 覆盖：v2.2 450 条（status {fmt_status(sn['status'])}）",
        f"- 执行跨度：{span:.0f} 秒（约 {span / 60:.1f} 分钟，主批次 {main_run['n']} 条，含批间冷却），run 数 {len(runs)}",
        "",
        "## 1. 关键指标（450 条同口径）",
        "",
        "| 指标 | v2.1 | v2.2 | 变化 |",
        "|---|---|---|---|",
        f"| done / degraded / failed | {fmt_status(so['status'])} | {fmt_status(sn['status'])} | - |",
        f"| 仅 other（证据不足桶） | {so['other_only']}（{so['other_only_pct']:.1f}%） | {sn['other_only']}（{sn['other_only_pct']:.1f}%） | {sn['other_only_pct'] - so['other_only_pct']:+.1f}pp |",
        f"| other 出现 / 主题总次数 | {so['other_occ']} / {so['theme_total']} | {sn['other_occ']} / {sn['theme_total']} | - |",
        f"| answer_quality | {so['aq']} | {sn['aq']} | {sn['aq'] - so['aq']:+d} |",
        f"| model_limits | {so['ml']} | {sn['ml']} | {sn['ml'] - so['ml']:+d} |",
        f"| missing_features | {so['mf']} | {sn['mf']} | {sn['mf'] - so['mf']:+d} |",
        f"| 引文命中 | {so['hits']}/{so['hit_n']}（{so['hit_pct']:.1f}%） | {sn['hits']}/{sn['hit_n']}（{sn['hit_pct']:.1f}%） | {sn['hit_pct'] - so['hit_pct']:+.1f}pp |",
        f"| 1–2★ 负面 | {so['consistency']['low_neg']}/{so['consistency']['low_n']}（{so['consistency']['low_pct']:.1f}%） | {sn['consistency']['low_neg']}/{sn['consistency']['low_n']}（{sn['consistency']['low_pct']:.1f}%） | {sn['consistency']['low_pct'] - so['consistency']['low_pct']:+.1f}pp |",
        f"| 4–5★ 正面 | {so['consistency']['high_pos']}/{so['consistency']['high_n']}（{so['consistency']['high_pct']:.1f}%） | {sn['consistency']['high_pos']}/{sn['consistency']['high_n']}（{sn['consistency']['high_pct']:.1f}%） | {sn['consistency']['high_pct'] - so['consistency']['high_pct']:+.1f}pp |",
        f"| 星级-情感冲突 | {so['consistency']['conflict']} | {sn['consistency']['conflict']} | {sn['consistency']['conflict'] - so['consistency']['conflict']:+d} |",
        "",
        "## 2. 主题排行（不含 other）",
        "",
        "> `other` 是**证据不足桶**，不参与主题排行；它表示原文没有足够具体的主题证据，不代表“没有意见”。",
        "",
        "| 排名 | 主题 | 出现次数 | 占 450 条 |",
        "|---|---|---|---|",
    ]
    for i, (t, n) in enumerate(ranked, 1):
        lines.append(f"| {i} | `{t}` | {n} | {n / 450 * 100:.1f}% |")
    lines += [
        f"| — | `other`（证据不足桶，不参与排行） | {sn['other_occ']} | {sn['other_occ'] / 450 * 100:.1f}% |",
        "",
        f"主题标注合计 {sum(n for _, n in ranked)} 次（不含 other），单条可多主题。",
        "",
        "## 3. 各产品主题差异（每产品 Top 5，不含 other）",
        "",
        "| 产品 | 条数 | other-only | Top 主题（次数） |",
        "|---|---|---|---|",
    ]
    for app in sorted({r["app"] for r in new}):
        rows = [r for r in new if r["app"] == app]
        c = Counter()
        for r in rows:
            c.update(t for t in themes_of(r) if t != "other")
        top = "、".join(f"`{t}` {n}" for t, n in c.most_common(5)) or "无"
        other_only = sum(1 for r in rows if themes_of(r) == ["other"])
        lines.append(f"| {app} | {len(rows)} | {other_only}（{other_only / len(rows) * 100:.1f}%） | {top} |")

    lines += [
        "",
        "## 4. severity 分布（非 skipped）",
        "",
        "| severity | v2.1 | v2.2 |",
        "|---|---|---|",
    ]
    for k in [0, 1, 2, 3, 4, 5, "null"]:
        lines.append(f"| {k} | {so['severity'].get(k, 0)} | {sn['severity'].get(k, 0)} |")

    lines += ["", "## 5. 降级 / 失败明细", ""]
    bad = [r for r in new if r["status"] in {"degraded", "failed"}]
    if not bad:
        lines.append("无。")
    else:
        lines += ["| id | 产品 | 星 | 状态 | 原因 | 原文摘要 |", "|---|---|---|---|---|---|"]
        for r in bad:
            content = (r["content"] or "").replace("\n", " ").replace("|", "／")[:80]
            lines.append(f"| {r['feedback_id']} | {r['app']} | {r['score']}★ | {r['status']} | `{r['error']}` | {content} |")

    skip = [r for r in new if r["status"] == "skipped"]
    lines += ["", f"规则跳过 {len(skip)} 条：{', '.join(str(r['feedback_id']) for r in skip) or '无'}（正文过短/无意义，不调用模型）。", ""]

    lines += [
        "## 6. 30 条抽读（同一套样本，v2.1 vs v2.2）",
        "",
        "样本与 `reports/p2_sample30.md` 完全一致；下表只列 v2.2 输出与判断。",
        "",
        "| # | id | 产品 | 星 | v2.2 主题/严重度 | 判断 | 说明 |",
        "|---|---|---|---|---|---|---|",
    ]
    sample_ids = [28, 32, 61, 78, 81, 85, 112, 136, 139, 150, 151, 171, 179,
                  213, 214, 219, 265, 278, 293, 300, 303, 326, 354, 374, 381,
                  384, 402, 416, 426, 431]
    by_id = {r["feedback_id"]: r for r in new}
    for i, fid in enumerate(sample_ids, 1):
        r = by_id.get(fid)
        if not r:
            lines.append(f"| {i} | {fid} | - | - | 缺失 | 待查 | - |")
            continue
        verdict, why = JUDGE30.get(fid, ("待人工判断", ""))
        lines.append(
            f"| {i} | {fid} | {r['app']} | {r['score']}★ | "
            f"`{','.join(themes_of(r)) or '-'}`/{r['severity']} | {verdict} | {why} |"
        )

    lines += [
        "",
        "## 7. 结论与局限",
        "",
        "- v2.2 只改了 `missing_features` 边界，其余主题规则不变；因此主题结构变化应主要出现在缺失类与 other。",
        "- `other` 始终单列为证据不足桶，不进入排行；排行只统计有具体证据的主题。",
        "- 星级-情感一致性只是启发式检查，不能替代人工阅读；本报告不含因果结论，也不能外推。",
        "- 450 条为三款产品的 Google Play 抽样，存在地区、语言、时间窗与平台偏差；不可外推到全部用户。",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "v21": {k: v for k, v in so.items() if k != "themes"},
        "v22": {k: v for k, v in sn.items() if k != "themes"},
        "v22_themes": dict(sn["themes"]),
        "span_s": round(span, 1),
        "report": str(REPORT),
    }, ensure_ascii=False, indent=2))
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
