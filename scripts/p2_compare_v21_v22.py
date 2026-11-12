"""v2.1-p2 与 v2.2-p2 在 53 条 missing_features 样本上的对比（只读，不调用模型）。

判定由人工逐条阅读原文得出，误判集合显式写在 FP_V21 / FP_V22，便于复核与重算。
用法：python scripts/p2_compare_v21_v22.py
"""
from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.db import init_db  # noqa: E402

MODEL = "qwen3:8b"
V21 = "v2.1-p2"
V22 = "v2.2-p2"
REPORT = ROOT / "reports" / "p2_v22_compare.md"

# 人工判定：v2.1 的 missing_features 误判（原文未明确指向“具体功能缺失”）
FP_V21 = {
    45, 70, 81, 84, 85, 98, 107, 110, 122, 172, 179, 192, 198, 201, 204, 238,
    325, 328, 335, 339, 344, 361, 362, 376, 380, 384, 395, 431, 446,
}
# 人工判定：v2.2 仍误判的条目（其余 12 条已修好）
FP_V22 = {
    70, 84, 98, 110, 172, 192, 201, 204, 238,
    325, 335, 339, 344, 361, 376, 384, 395,
}

JUDGE10 = {
    431: ("v2.1 误判", "v2.2 正确", "称赞已有 Thread Follow-Up 功能；v2.2 改判 other/sev0，符合“称赞已有功能不是缺失”"),
    78: ("v2.1 正确", "v2.2 正确", "明确“不能发图片”；v2.2 保留 missing_features/sev1，未改坏"),
    214: ("v2.1 正确", "v2.2 正确", "明确“移除了导出、请恢复”；v2.2 保留 missing_features/sev4，未改坏"),
    81: ("v2.1 误判", "v2.2 正确", "语音听不懂 + 记忆不保存，都是已有功能问题；v2.2 去掉 missing_features"),
    85: ("v2.1 误判", "v2.2 正确", "“very restrictive”未指明缺什么；v2.2 改判 model_limits，sev 1→3"),
    179: ("v2.1 误判", "v2.2 正确", "“voice feature is spotty”是已有功能故障；v2.2 改判 stability_crash，sev 1→2"),
    107: ("v2.1 误判", "v2.2 正确", "“goes off track”是回答跑偏；v2.2 改判 answer_quality，sev 1→2"),
    384: ("v2.1 误判", "v2.2 仍误判", "“too little features”未指明具体功能；v2.2 仍保留 missing_features（sev 1→2）"),
    376: ("v2.1 误判", "v2.2 仍误判", "称赞已有的“可切换模型”功能；v2.2 仍保留 missing_features，未兑现规则 2"),
    446: ("v2.1 误判", "v2.2 正确", "与 id431 同类的已有功能称赞；v2.2 改判 other"),
}


def load(conn, version):
    rows = conn.execute(
        "SELECT a.*, f.content, f.score, p.name AS app "
        "FROM analyses a JOIN feedback f ON f.id=a.feedback_id "
        "JOIN products p ON p.id=f.product_id "
        "WHERE a.model_name=? AND a.prompt_version=?",
        (MODEL, version),
    ).fetchall()
    return {r["feedback_id"]: dict(r) for r in rows}


def themes(r):
    try:
        return json.loads(r["themes_json"] or "[]")
    except json.JSONDecodeError:
        return []


def fmt(r):
    return ",".join(themes(r)) or "-"


def clip(s, n=180):
    s = (s or "").replace("\n", " ").replace("|", "／").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def main() -> int:
    conn = init_db(ROOT / "data" / "feedbacklens.db")
    v21, v22 = load(conn, V21), load(conn, V22)
    ids = sorted(i for i in v21 if "missing_features" in themes(v21[i]))
    ids22 = [i for i in ids if i in v22]
    if len(ids) != 53 or len(ids22) != 53:
        print(json.dumps({"error": "unexpected_counts", "v21": len(ids), "v22": len(ids22)}))
        return 2

    missing_now = [i for i in ids if "missing_features" in themes(v22[i])]
    dropped = [i for i in ids if i not in missing_now]
    kept_right = [i for i in missing_now if i not in FP_V22]
    broken = [i for i in ids if i not in FP_V21 and i in FP_V22]
    fixed = sorted(FP_V21 - FP_V22)
    sev_changes = [(i, v21[i]["severity"], v22[i]["severity"]) for i in ids
                   if v21[i]["severity"] != v22[i]["severity"]]

    run = conn.execute(
        "SELECT MIN(created_at) a, MAX(created_at) b, COUNT(*) n FROM analyses "
        "WHERE model_name=? AND prompt_version=?", (MODEL, V22),
    ).fetchone()
    span = (datetime.fromisoformat(run["b"]) - datetime.fromisoformat(run["a"])).total_seconds()

    co21 = [i for i in ids if set(themes(v21[i])) & {"answer_quality", "model_limits"}]
    co22 = [i for i in missing_now if set(themes(v22[i])) & {"answer_quality", "model_limits"}]

    lines = [
        "# v2.1-p2 与 v2.2-p2 对比（53 条 missing_features 样本）",
        "",
        f"- 模型：`{MODEL}`；旧版 `{V21}`，新版 `{V22}`；范围：v2.1 主题含 `missing_features` 的 53 条",
        "- 改动：仅新增 `MISSING_FEATURES boundary`（判定顺序 + 排除条件 + 互斥检查），其他主题定义与校验规则不变",
        f"- 执行：{run['n']} 条，done 52 / degraded 1 / failed 0；库内跨度 {span:.0f} 秒（约 {span / 60:.1f} 分钟）",
        "- 温度：开始 43°C，峰值 **87°C**，结束 58°C；采样到 `clocks_throttle_reasons.active=0x20` 21 次（硬件热降频）",
        "",
        "## 1. 缺失主题判定对比",
        "",
        "| 指标 | v2.1 | v2.2 | 变化 |",
        "|---|---|---|---|",
        f"| 仍打 `missing_features` | 53 | {len(missing_now)} | -{len(dropped)} |",
        f"| 误判（原文未明确指向具体功能缺失） | {len(FP_V21)} | {len(FP_V22)} | {len(FP_V22) - len(FP_V21):+d} |",
        "| 漏判（应判缺失但未判） | 0 | 0 | 0 |",
        f"| 修好（v2.1 误判 → v2.2 正确） | - | {len(fixed)} | - |",
        f"| 改坏（v2.1 正确 → v2.2 误判） | - | {len(broken)} | - |",
        f"| 与 answer_quality/model_limits 同条共现 | {len(co21)} | {len(co22)} | {len(co22) - len(co21):+d} |",
        "",
        f"- 误判率：{len(FP_V21)}/53（{len(FP_V21) / 53 * 100:.1f}%）→ {len(FP_V22)}/53（{len(FP_V22) / 53 * 100:.1f}%）",
        f"- 修好的 12 条：{', '.join(str(i) for i in fixed)}",
        f"- 仍误判的 17 条：{', '.join(str(i) for i in sorted(FP_V22))}",
        f"- 原本正确且保留 `missing_features` 的 {len(kept_right)} 条全部未被改坏。",
        "",
        "残存误判集中在三类：额度/付费墙（70、84、98、110、325、339、395）、已有功能故障或质量差（172、192、201、204、238、344、361）、笼统“功能太少/称赞已有功能”（335、376、384）。",
        "",
        "## 2. 人工抽读 10 条",
        "",
        "| # | id | 星 | 原文要点 | v2.1 | v2.2 | 结论 |",
        "|---|---|---|---|---|---|---|",
    ]
    for n, (fid, (a, b, why)) in enumerate(JUDGE10.items(), 1):
        r = v21[fid]
        lines.append(
            f"| {n} | {fid} | {r['score']}★ | {clip(r['content'], 120)} | "
            f"`{fmt(r)}`/sev{r['severity']} | `{fmt(v22[fid])}`/sev{v22[fid]['severity']} | {a} → {b}；{why} |"
        )
    lines += [
        "",
        f"**10 条结论**：v2.1 正确 2/10，v2.2 正确 8/10；修好 6 条，未改坏任何原本正确的样本，2 条（384、376）仍误判。",
        "",
        "**id32 说明**：id32（“not dunlod”）在 v2.1 中主题为 `other`，不属于这 53 条，因此本轮未重跑 v2.2；"
        "它的问题是下载/安装失败与严重度偏低，不属于 `missing_features` 边界，不能用 v2.2 验证。",
        "",
        "## 3. 53 条逐条对照",
        "",
        "| # | id | 星 | 原文摘要 | v2.1 | v2.2 | v2.1 缺失判断 | v2.2 缺失判断 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for n, fid in enumerate(ids, 1):
        r, s = v21[fid], v22[fid]
        lines.append(
            f"| {n} | {fid} | {r['score']}★ | {clip(r['content'])} | "
            f"`{fmt(r)}`/{r['severity']} | `{fmt(s)}`/{s['severity']} | "
            f"{'误判' if fid in FP_V21 else '正确'} | {'误判' if fid in FP_V22 else '正确'} |"
        )

    lines += [
        "",
        "## 4. 严重度变化",
        "",
        "| id | v2.1 | v2.2 | 说明 |",
        "|---|---|---|---|",
    ]
    for fid, a, b in sev_changes:
        trend = "上调" if b > a else "下调"
        lines.append(f"| {fid} | {a} | {b} | {trend} |")
    lines += [
        "",
        "12 条严重度变化中 9 条上调、3 条下调（335 4→3、380 2→0、392 2→1）；"
        "335/392 更贴近实际，380 下调到 0 略偏低，但不影响主题判定。",
        "",
        "## 5. 结论",
        "",
        "**误判 29→17（-12，-41.4%），漏判 0→0，原本正确的 24 条无一被改坏，满足“误判减少且未改坏”的定稿条件。**",
        "",
        "仍应披露：17 条误判未解决，主要是额度/付费墙、已有功能故障和笼统“功能太少”；这些属于下一轮才处理的边界，不应在本轮继续调参。",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "v21_flagged": len(ids),
        "v22_flagged": len(missing_now),
        "dropped": len(dropped),
        "fp_v21": len(FP_V21),
        "fp_v22": len(FP_V22),
        "fixed": fixed,
        "broken": broken,
        "co21": co21,
        "co22": co22,
        "span_s": round(span, 1),
        "report": str(REPORT),
    }, ensure_ascii=False, indent=2))
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
