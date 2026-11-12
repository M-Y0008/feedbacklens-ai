"""一键收尾（需在沙箱外终端运行）：

1) 删除 eval_runs 中重复的 eval-20261109-195129 一行（保留 id 较小的一行）；
2) 把 v2.3 的 60 条结果写入 analyses（只新增 prompt_version='v2.3' 的行，不动 v2.2）。

写前自动备份数据库；重复执行安全（幂等）。
用法：
    python scripts/p2_finalize_v23.py
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.batch import _save_result  # noqa: E402

DB = Path(os.environ.get("FEEDBACKLENS_DB") or (ROOT / "data" / "feedbacklens.db"))
JSONL = ROOT / "reports" / "v23_results_60.jsonl"
DUP_RUN_ID = "eval-20261109-195129"
RUN_ID = "v23-60-20261109-204825"


class _Backend:
    model = "qwen3:8b"


def backup() -> Path:
    bak = DB.with_name(DB.name + "." + datetime.now().strftime("%Y%m%d%H%M%S") + ".bak")
    src = sqlite3.connect(DB)
    dst = sqlite3.connect(bak)
    src.backup(dst)
    dst.close()
    src.close()
    return bak


def dedup(conn: sqlite3.Connection) -> None:
    rows = [dict(r) for r in conn.execute(
        "SELECT * FROM eval_runs WHERE run_id=? ORDER BY id", (DUP_RUN_ID,))]
    if len(rows) <= 1:
        print(f"[dedup] 已是 {len(rows)} 行，跳过")
        return
    if len(rows) != 2:
        print(f"[dedup] 预期 2 行、实际 {len(rows)} 行，跳过，请人工检查")
        return
    sig = lambda r: (r["sample_size"], r["labeler"], r["metrics_json"],
                     r["confusion_json"], r["model_name"], r["prompt_version"])
    if sig(rows[0]) != sig(rows[1]):
        print("[dedup] 两行内容不同，跳过，请人工检查")
        return
    conn.execute("DELETE FROM eval_runs WHERE id=?", (rows[1]["id"],))
    conn.commit()
    left = conn.execute(
        "SELECT COUNT(*) FROM eval_runs WHERE run_id=?", (DUP_RUN_ID,)).fetchone()[0]
    print(f"[dedup] 已删除 id={rows[1]['id']}，剩余 {left} 行")


def apply_v23(conn: sqlite3.Connection) -> None:
    recs = [json.loads(l) for l in JSONL.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"[v2.3] 待写入 {len(recs)} 条，run_id={RUN_ID}")
    cfg = {"prompt_version": "v2.3"}
    for n, rec in enumerate(recs, 1):
        item = {"id": rec["feedback_id"], "review_id": rec["review_id"],
                "app": rec["app"], "score": rec["score"], "content": rec["content"]}
        r = {k: rec[k] for k in (
            "themes", "sentiment", "intent", "pain_point", "feature_request",
            "severity", "evidence_quote", "confidence", "needs_human_review",
            "is_valid", "raw_output", "error", "status")}
        _save_result(conn, item, RUN_ID, _Backend(), cfg, r)
        if n % 15 == 0:
            conn.commit()
    conn.commit()
    n23 = conn.execute(
        "SELECT COUNT(*) FROM analyses WHERE prompt_version='v2.3'").fetchone()[0]
    n22 = conn.execute(
        "SELECT COUNT(*) FROM analyses WHERE prompt_version='v2.2-p2'").fetchone()[0]
    print(f"[v2.3] 写入完成：v2.3={n23} 行，v2.2-p2={n22} 行（应保持 450）")
    print("[v2.3] 若 v2.2-p2 不是 450，请立即用备份恢复并检查。")


def main() -> int:
    if not DB.exists():
        print("找不到数据库：", DB)
        return 1
    print("数据库：", DB)
    print("备份：", backup())
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    dedup(conn)
    apply_v23(conn)
    conn.close()
    print("\n完成。刷新 Streamlit「AI 质量评估」页，版本选择 v2.3 即可看到对比指标。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
