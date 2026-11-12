"""从 450 条 v2.1 结果中抽取 30 条人工核查样本（只读）。

抽样规则（固定 seed=2026，可复现）：
- 每款产品随机 8 条（覆盖三款产品）
- 每款产品再从主题含 missing_features 的样本中取 2 条（不足则用随机样本补齐）
- 去重后共 30 条，写入 reports/p2_sample30.md
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.db import init_db  # noqa: E402

SEED = 2026
PER_APP_RANDOM = 8
PER_APP_MISSING = 2
OUT = ROOT / "reports" / "p2_sample30.md"


def themes_of(row):
    try:
        t = json.loads(row["themes_json"] or "[]")
    except json.JSONDecodeError:
        return []
    return t if isinstance(t, list) else []


def main() -> int:
    conn = init_db(ROOT / "data" / "feedbacklens.db")
    rows = [dict(r) for r in conn.execute(
        "SELECT a.*, f.content, f.score, p.name AS app "
        "FROM analyses a JOIN feedback f ON f.id=a.feedback_id "
        "JOIN products p ON p.id=f.product_id "
        "WHERE a.model_name='qwen3:8b' AND a.prompt_version='v2.1-p2' "
        "AND a.status IN ('done','degraded') ORDER BY f.id"
    ).fetchall()]
    apps = sorted({r["app"] for r in rows})
    rng = random.Random(SEED)
    picked: list[dict] = []
    picked_ids: set[int] = set()

    for app in apps:
        pool = [r for r in rows if r["app"] == app]
        for r in rng.sample(pool, min(PER_APP_RANDOM, len(pool))):
            picked.append(r)
            picked_ids.add(r["feedback_id"])

    for app in apps:
        pool = [r for r in rows if r["app"] == app and "missing_features" in themes_of(r)
                and r["feedback_id"] not in picked_ids]
        for r in rng.sample(pool, min(PER_APP_MISSING, len(pool))):
            picked.append(r)
            picked_ids.add(r["feedback_id"])

    picked.sort(key=lambda r: (r["app"], r["feedback_id"]))

    lines = [
        "# v2.1-p2 人工抽读 30 条（原始记录）",
        "",
        f"- 抽样：seed={SEED}；每产品随机 {PER_APP_RANDOM} 条 + missing_features {PER_APP_MISSING} 条",
        f"- 实际样本：{len(picked)} 条；产品分布：{ {a: sum(1 for r in picked if r['app']==a) for a in apps} }",
        f"- 主题含 missing_features 的样本：{sum(1 for r in picked if 'missing_features' in themes_of(r))} 条",
        "",
    ]
    for i, r in enumerate(picked, 1):
        content = (r["content"] or "").replace("\n", " ")
        lines += [
            f"## {i}. id={r['feedback_id']} / {r['app']} / {r['score']}★ / {r['status']}",
            "",
            f"- 原文：{content[:500]}",
            f"- 主题：`{','.join(themes_of(r))}`",
            f"- 情感 / 意图 / 严重度：`{r['sentiment']}` / `{r['intent']}` / `{r['severity']}`",
            f"- pain_point：`{r['pain_point']}`",
            f"- feature_request：`{r['feature_request']}`",
            f"- 引文：「{r['evidence_quote'] or ''}」",
            f"- confidence / needs_human_review：`{r['confidence']}` / `{r['needs_human_review']}`",
            "",
        ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "sample_size": len(picked),
        "by_app": {a: sum(1 for r in picked if r["app"] == a) for a in apps},
        "missing_features_rows": sum(1 for r in picked if "missing_features" in themes_of(r)),
        "ids": [r["feedback_id"] for r in picked],
        "report": str(OUT),
    }, ensure_ascii=False, indent=2))
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
