"""v2.0-p2 与 v2.1-p2 并排对比（只读，不调用模型）。

范围：v2.0 主题仅 other 或含 model_limits 的 312 条做定向新旧对比；
450 条做 v2.0 全量 vs v2.1 全量的同口径对比。
用法：python scripts/p2_compare_v20_v21.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.analysis import THEMES, normalize_ws  # noqa: E402
from core.db import init_db  # noqa: E402

MODEL = "qwen3:8b"
OLD = "v2.0-p2"
NEW = "v2.1-p2"
REPORT = ROOT / "reports" / "p2_v21_compare.md"

STOP = {
    "the", "and", "for", "with", "this", "that", "app", "apps", "very", "not",
    "but", "from", "have", "has", "are", "was", "were", "you", "your", "its",
    "it", "is", "of", "to", "in", "on", "can", "cannot", "does", "doesn", "when",
    "always", "never", "more", "much", "some", "any", "all", "too", "also",
}


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
    neg_low = sum(1 for r in low if r["sentiment"] == "negative")
    pos_high = sum(1 for r in high if r["sentiment"] == "positive")
    conflict = sum(
        1 for r in rows
        if (r["score"] <= 2 and r["sentiment"] == "positive")
        or (r["score"] >= 4 and r["sentiment"] == "negative")
    )
    return {
        "low_n": len(low),
        "low_neg": neg_low,
        "low_neg_pct": neg_low / len(low) * 100 if low else 0,
        "high_n": len(high),
        "high_pos": pos_high,
        "high_pos_pct": pos_high / len(high) * 100 if high else 0,
        "conflict": conflict,
    }


def pain_unsupported(rows):
    out = []
    for r in rows:
        pp = (r.get("pain_point") or "").strip()
        if not pp:
            continue
        words = [w for w in re.findall(r"[a-z]{4,}", pp.lower()) if w not in STOP]
        content = (r.get("content") or "").lower()
        if words and not any(w in content for w in words):
            out.append(r)
    return out


def summarize(rows):
    themes = Counter()
    other_only = 0
    for r in rows:
        t = themes_of(r)
        if t == ["other"]:
            other_only += 1
        themes.update(t)
    analyzed = [r for r in rows if r["status"] != "skipped"]
    status = Counter(r["status"] for r in rows)
    sev = Counter(("null" if r["severity"] is None else r["severity"]) for r in analyzed)
    hits = sum(1 for r in analyzed if quote_ok(r))
    return {
        "n": len(rows),
        "status": dict(status),
        "other_only": other_only,
        "other_only_pct": other_only / len(rows) * 100 if rows else 0,
        "other_occurrence": themes.get("other", 0),
        "theme_occurrence": sum(themes.values()),
        "themes": themes,
        "answer_quality": themes.get("answer_quality", 0),
        "model_limits": themes.get("model_limits", 0),
        "quote_hit": hits,
        "quote_hit_pct": hits / len(analyzed) * 100 if analyzed else 0,
        "quote_hit_n": len(analyzed),
        "severity": sev,
        "consistency": consistency(analyzed),
        "unsupported_pp": len(pain_unsupported(analyzed)),
        "unsupported_examples": pain_unsupported(analyzed)[:5],
    }


def pick_samples(old_rows, new_rows, n_other=6, n_limits=4):
    by_id_new = {r["feedback_id"]: r for r in new_rows}
    groups = {"other": [], "limits": []}
    for r in old_rows:
        t = themes_of(r)
        if t == ["other"]:
            groups["other"].append(r)
        elif "model_limits" in t:
            groups["limits"].append(r)

    def spaced(rows, k):
        rows = sorted(rows, key=lambda r: (r["app"], r["score"], r["feedback_id"]))
        if len(rows) <= k:
            return rows
        idx = sorted({round(i * (len(rows) - 1) / (k - 1)) for i in range(k)})
        return [rows[i] for i in idx]

    picked = spaced(groups["other"], n_other) + spaced(groups["limits"], n_limits)
    picked.sort(key=lambda r: (r["app"], r["score"], r["feedback_id"]))
    return [(r, by_id_new.get(r["feedback_id"])) for r in picked]


def fmt_themes(r):
    return ",".join(themes_of(r)) or "-"


def main() -> int:
    conn = init_db(ROOT / "data" / "feedbacklens.db")
    old_all = load(conn, OLD)
    new_all = load(conn, NEW)
    if len(old_all) != 450 or len(new_all) not in (312, 450):
        print(json.dumps({"error": "unexpected_row_counts", "old": len(old_all), "new": len(new_all)}))
        return 2

    affected_ids = set()
    for r in old_all:
        t = themes_of(r)
        if t == ["other"] or "model_limits" in t:
            affected_ids.add(r["feedback_id"])

    old_sub = [r for r in old_all if r["feedback_id"] in affected_ids]
    new_sub = [r for r in new_all if r["feedback_id"] in affected_ids]
    if len(new_sub) != len(old_sub):
        print(json.dumps({"error": "subset_mismatch", "old": len(old_sub), "new": len(new_sub)}))
        return 2

    s_old = summarize(old_sub)
    s_new = summarize(new_sub)
    s_old_all = summarize(old_all)
    s_all = summarize(new_all)

    # 若 450 条已全部升级到 v2.1，则整体构成为同口径；否则为混合口径
    if len(new_all) >= 450:
        merged = new_all
        coverage = "450 条全部为 v2.1（同口径）"
    else:
        merged = [r for r in old_all if r["feedback_id"] not in affected_ids] + new_sub
        coverage = f"{len(new_sub)} 条 v2.1 + {450 - len(new_sub)} 条 v2.0（混合口径）"
    s_merged = summarize(merged)

    # v2.1 两段实际耗时（按 created_at 跨度求和）
    spans = []
    for row in conn.execute(
        "SELECT run_id, MIN(created_at) AS a, MAX(created_at) AS b "
        "FROM analyses WHERE model_name=? AND prompt_version=? GROUP BY run_id",
        (MODEL, NEW),
    ):
        from datetime import datetime
        a = datetime.fromisoformat(row["a"])
        b = datetime.fromisoformat(row["b"])
        spans.append((row["run_id"], (b - a).total_seconds()))
    total_span = sum(s for _, s in spans)

    def fmt_status(s):
        out = f"done {s.get('done', 0)} / degraded {s.get('degraded', 0)} / failed {s.get('failed', 0)}"
        if s.get("skipped"):
            out += f" / skipped {s['skipped']}"
        return out


    def top_rows(counter, k=12):
        return [(t, counter.get(t, 0)) for t in THEMES if counter.get(t, 0)][:k]

    lines = [
        "# v2.0-p2 与 v2.1-p2 并排对比（450 条同口径）",
        "",
        f"- 模型：`{MODEL}`；旧版 `{OLD}`，新版 `{NEW}`；样本：450 条全量",
        f"- 覆盖：{coverage}",
        "- 定向对比范围：旧版主题仅 `other` 或含 `model_limits` 的 312 条",
        "- 口径：v2.1 主题只依据原文证据；不要求为降低 other 而强行归类",
        "",
        "## 1. 核心指标对比（312 条同一批评论）",
        "",
        "| 指标 | v2.0 | v2.1 | 变化 |",
        "|---|---|---|---|",
        f"| 状态 done / degraded / failed | {fmt_status(s_old['status'])} | {fmt_status(s_new['status'])} | - |",
        f"| 仅 other 的条数（占本批） | {s_old['other_only']}（{s_old['other_only_pct']:.1f}%） | {s_new['other_only']}（{s_new['other_only_pct']:.1f}%） | {s_new['other_only_pct'] - s_old['other_only_pct']:+.1f}pp |",
        f"| other 出现次数 / 主题总次数 | {s_old['other_occurrence']} / {s_old['theme_occurrence']} | {s_new['other_occurrence']} / {s_new['theme_occurrence']} | - |",
        f"| answer_quality 命中 | {s_old['answer_quality']} | {s_new['answer_quality']} | {s_new['answer_quality'] - s_old['answer_quality']:+d} |",
        f"| model_limits 命中 | {s_old['model_limits']} | {s_new['model_limits']} | {s_new['model_limits'] - s_old['model_limits']:+d} |",
        f"| 引文命中率 | {s_old['quote_hit_pct']:.1f}% | {s_new['quote_hit_pct']:.1f}% | {s_new['quote_hit_pct'] - s_old['quote_hit_pct']:+.1f}pp |",
        f"| 1–2★ 负面占比 | {s_old['consistency']['low_neg']}/{s_old['consistency']['low_n']}（{s_old['consistency']['low_neg_pct']:.1f}%） | {s_new['consistency']['low_neg']}/{s_new['consistency']['low_n']}（{s_new['consistency']['low_neg_pct']:.1f}%） | {s_new['consistency']['low_neg_pct'] - s_old['consistency']['low_neg_pct']:+.1f}pp |",
        f"| 4–5★ 正面占比 | {s_old['consistency']['high_pos']}/{s_old['consistency']['high_n']}（{s_old['consistency']['high_pos_pct']:.1f}%） | {s_new['consistency']['high_pos']}/{s_new['consistency']['high_n']}（{s_new['consistency']['high_pos_pct']:.1f}%） | {s_new['consistency']['high_pos_pct'] - s_old['consistency']['high_pos_pct']:+.1f}pp |",
        f"| 星级-情感冲突条数 | {s_old['consistency']['conflict']} | {s_new['consistency']['conflict']} | {s_new['consistency']['conflict'] - s_old['consistency']['conflict']:+d} |",
        f"| pain_point 词面疑似无原文依据 | {s_old['unsupported_pp']} | {s_new['unsupported_pp']} | {s_new['unsupported_pp'] - s_old['unsupported_pp']:+d} |",
        "",
        "## 2. 主题分布（本批 312 条）",
        "",
        "| 主题 | v2.0 | v2.1 |",
        "|---|---|---|",
    ]
    for t, n in top_rows(s_old["themes"]):
        lines.append(f"| {t} | {n} | {s_new['themes'].get(t, 0)} |")
    for t, n in top_rows(s_new["themes"]):
        if not s_old["themes"].get(t):
            lines.append(f"| {t} | 0 | {n} |")

    lines += [
        "",
        "## 3. 450 条同口径整体对比（v2.0 全量 vs v2.1 全量）",
        "",
        f"覆盖：{coverage}",
        "",
        "| 指标 | v2.0（450） | v2.1（450） | 变化 |",
        "|---|---|---|---|",
        f"| 状态 done / degraded / failed | {fmt_status(s_old_all['status'])} | {fmt_status(s_all['status'])} | - |",
        f"| 仅 other 的条数 | {s_old_all['other_only']}（{s_old_all['other_only_pct']:.1f}%） | {s_all['other_only']}（{s_all['other_only_pct']:.1f}%） | {s_all['other_only_pct'] - s_old_all['other_only_pct']:+.1f}pp |",
        f"| other 出现次数 / 主题总次数 | {s_old_all['other_occurrence']} / {s_old_all['theme_occurrence']} | {s_all['other_occurrence']} / {s_all['theme_occurrence']} | - |",
        f"| answer_quality 命中 | {s_old_all['answer_quality']} | {s_all['answer_quality']} | {s_all['answer_quality'] - s_old_all['answer_quality']:+d} |",
        f"| model_limits 命中 | {s_old_all['model_limits']} | {s_all['model_limits']} | {s_all['model_limits'] - s_old_all['model_limits']:+d} |",
        f"| 引文命中率 | {s_old_all['quote_hit']}/{s_old_all['quote_hit_n']}（{s_old_all['quote_hit_pct']:.1f}%） | {s_all['quote_hit']}/{s_all['quote_hit_n']}（{s_all['quote_hit_pct']:.1f}%） | {s_all['quote_hit_pct'] - s_old_all['quote_hit_pct']:+.1f}pp |",
        f"| 1–2★ 负面占比 | {s_old_all['consistency']['low_neg']}/{s_old_all['consistency']['low_n']}（{s_old_all['consistency']['low_neg_pct']:.1f}%） | {s_all['consistency']['low_neg']}/{s_all['consistency']['low_n']}（{s_all['consistency']['low_neg_pct']:.1f}%） | {s_all['consistency']['low_neg_pct'] - s_old_all['consistency']['low_neg_pct']:+.1f}pp |",
        f"| 4–5★ 正面占比 | {s_old_all['consistency']['high_pos']}/{s_old_all['consistency']['high_n']}（{s_old_all['consistency']['high_pos_pct']:.1f}%） | {s_all['consistency']['high_pos']}/{s_all['consistency']['high_n']}（{s_all['consistency']['high_pos_pct']:.1f}%） | {s_all['consistency']['high_pos_pct'] - s_old_all['consistency']['high_pos_pct']:+.1f}pp |",
        f"| 星级-情感冲突条数 | {s_old_all['consistency']['conflict']} | {s_all['consistency']['conflict']} | {s_all['consistency']['conflict'] - s_old_all['consistency']['conflict']:+d} |",
        f"| pain_point 词面疑似无原文依据 | {s_old_all['unsupported_pp']} | {s_all['unsupported_pp']} | {s_all['unsupported_pp'] - s_old_all['unsupported_pp']:+d} |",
        "",
        "### 3.1 主题出现次数（单条可多主题）",
        "",
        "| 主题 | v2.0 | v2.1 | 变化 |",
        "|---|---|---|---|",
    ]
    for t in sorted(
        THEMES,
        key=lambda x: (-s_all["themes"].get(x, 0), -s_old_all["themes"].get(x, 0)),
    ):
        o, n = s_old_all["themes"].get(t, 0), s_all["themes"].get(t, 0)
        if o or n:
            lines.append(f"| {t} | {o} | {n} | {n - o:+d} |")

    lines += [
        f"| **other（仅 other 的评论）** | {s_old_all['other_only']} 条 | {s_all['other_only']} 条 | {s_all['other_only'] - s_old_all['other_only']:+d} |",
        "",
        "## 4. severity 分布",
        "",
        "| severity | v2.0(450) | v2.1(450) | v2.0(312) | v2.1(312) |",
        "|---|---|---|---|---|",
    ]
    for k in [0, 1, 2, 3, 4, 5, "null"]:
        lines.append(
            f"| {k} | {s_old_all['severity'].get(k, 0)} | {s_all['severity'].get(k, 0)} "
            f"| {s_old['severity'].get(k, 0)} | {s_new['severity'].get(k, 0)} |"
        )
    lines += [
        "",
        "## 5. 耗时",
        "",
        f"- v2.1 各段 created_at 跨度合计 {total_span:.0f} 秒（{total_span / 60:.1f} 分钟），其中含 id97 一次 930.3 秒异常挂起；",
        "  剔除该异常后单条约 1.9–2.2 秒（138 条批次中位 2.21 秒、最长 9.91 秒）。",
        "- v2.0 全量 450 条此前为 15.2 分钟。",
        "",
        "## 6. 抽读样本（旧 → 新）",
        "",
    ]
    for i, (o, n) in enumerate(pick_samples(old_sub, new_sub), 1):
        if n is None:
            continue
        lines += [
            f"### {i}. feedback_id={o['feedback_id']} / {o['app']} / {o['score']}★",
            "",
            f"- 原文：{(o['content'] or '').replace(chr(10), ' ')[:500]}",
            f"- v2.0：themes=`{fmt_themes(o)}` / sentiment=`{o['sentiment']}` / severity=`{o['severity']}` / pain_point=`{o['pain_point'] or 'null'}` / quote=「{o['evidence_quote'] or ''}」",
            f"- v2.1：themes=`{fmt_themes(n)}` / sentiment=`{n['sentiment']}` / severity=`{n['severity']}` / pain_point=`{n['pain_point'] or 'null'}` / quote=「{n['evidence_quote'] or ''}」",
            "",
        ]

    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "old": {k: v for k, v in s_old.items() if k not in {"themes", "unsupported_examples"}},
        "new": {k: v for k, v in s_new.items() if k not in {"themes", "unsupported_examples"}},
        "all_old": {k: v for k, v in s_old_all.items() if k not in {"themes", "unsupported_examples"}},
        "all_new": {k: v for k, v in s_all.items() if k not in {"themes", "unsupported_examples"}},
        "merged": {k: v for k, v in s_merged.items() if k not in {"themes", "unsupported_examples"}},
        "coverage": coverage,
        "old_themes": dict(s_old["themes"]),
        "new_themes": dict(s_new["themes"]),
        "all_old_themes": dict(s_old_all["themes"]),
        "all_new_themes": dict(s_all["themes"]),
        "merged_themes": dict(s_merged["themes"]),
        "v21_total_span_s": round(total_span, 1),
        "new_unsupported_examples": [
            {"feedback_id": r["feedback_id"], "content": (r["content"] or "")[:160],
             "pain_point": r["pain_point"], "quote": r["evidence_quote"]}
            for r in s_new["unsupported_examples"]
        ],
        "report": str(REPORT),
    }, ensure_ascii=False, indent=2))
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
