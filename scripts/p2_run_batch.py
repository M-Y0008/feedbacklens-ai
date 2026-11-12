"""P2 小批量真实验证：对已入库评论跑单条分析引擎（默认 30 条），产出中文报告。

分批 + 批间冷却：每批默认 60 条，批间强制冷却 120 秒并打印 GPU 温度，
并发始终为 1 条，避免长时间满载导致笔记本过热。

用法：
  python scripts/p2_run_batch.py --limit 30
  python scripts/p2_run_batch.py --model qwen3:8b
  python scripts/p2_run_batch.py --limit 450 --batch-size 60 --cooldown-seconds 120
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.batch import (  # noqa: E402
    run_batch, select_affected, select_remaining, select_theme, select_trial,
)
from core.db import backup_if_exists, init_db  # noqa: E402
from core.llm import get_backend, list_ollama_models, load_config  # noqa: E402


def build_metrics(stats: Counter, details: list[dict]) -> dict:
    done = stats.get("done", 0)
    degraded = stats.get("degraded", 0)
    failed = stats.get("failed", 0)
    called = done + degraded + failed
    schema_complete = sum(
        1 for d in details
        if d["status"] != "skipped" and not d["missing_fields"] and not d["invalid_fields"]
    )
    quote_hits = sum(1 for d in details if d.get("quote_ok"))
    times = [d["seconds"] for d in details if d["status"] != "skipped"]
    failure_types: Counter = Counter()
    missing_counter: Counter = Counter()
    flags_counter: Counter = Counter()
    for d in details:
        if d["status"] == "failed":
            err = d["error"]
            key = ":".join(err.split(":")[:2]) if err.startswith("call:") else err.split(":")[0]
            failure_types[key] += 1
        elif d["status"] == "degraded":
            for part in (d["error"] or "").split("; "):
                if not part:
                    continue
                if part.startswith("missing:"):
                    missing_counter[part] += 1
                elif part.startswith("filled:") or part.startswith("flag:"):
                    continue
                else:
                    failure_types["degraded:" + part.split(":")[0]] += 1
        for f in d.get("flags") or []:
            flags_counter[f.split("(")[0]] += 1

    return {
        "attempted": called,
        "done": done,
        "degraded": degraded,
        "failed": failed,
        "rule_skipped": stats.get("skipped", 0),
        "resume_skipped": stats.get("resume_skipped", 0),
        "max_attempts_skipped": stats.get("max_attempts_skipped", 0),
        "success_rate": f"{done / called * 100:.0f}%" if called else "n/a",
        "usable_rate": f"{(done + degraded) / called * 100:.0f}%" if called else "n/a",
        "schema_complete_rate": f"{schema_complete / called * 100:.0f}%" if called else "n/a",
        "quote_hit_rate": f"{quote_hits / called * 100:.0f}%" if called else "n/a",
        "latency_avg_s": round(statistics.mean(times), 2) if times else None,
        "latency_median_s": round(statistics.median(times), 2) if times else None,
        "latency_max_s": round(max(times), 2) if times else None,
        "failure_types": dict(failure_types),
        "missing_fields": dict(missing_counter),
        "flags": dict(flags_counter),
    }


def _clip(s, n: int) -> str:
    s = (s or "").replace("\n", " ").replace("|", "／").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


THROTTLE_MASK = 0xEC  # 0x4 功率 / 0x8 硬件 / 0x20 软件热 / 0x40 硬件热 / 0x80 功率制动


def gpu_state() -> tuple[int, int, int, str] | None:
    """返回 (温度, 利用率, 显存, 降频位)；不可用时返回 None。"""
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=temperature.gpu,utilization.gpu,memory.used,clocks_throttle_reasons.active",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
        if out.returncode != 0 or not out.stdout.strip():
            return None
        parts = [p.strip() for p in out.stdout.strip().splitlines()[0].split(",")]
        if len(parts) < 4:
            return None
        return int(parts[0]), int(parts[1]), int(parts[2]), parts[3]
    except Exception:
        return None


def thermal_unsafe() -> bool:
    """≥90°C，或温度 ≥85°C 且出现热/功率降频位时，判定为不安全。"""
    g = gpu_state()
    if not g:
        return False
    temp, _, _, throttle = g
    try:
        bits = int(throttle, 16) & THROTTLE_MASK
    except ValueError:
        bits = 0
    return temp >= 90 or (bits != 0 and temp >= 85)


def gpu_thermal() -> str:
    """读取当前 GPU 温度/利用率/显存/降频；无 nvidia-smi 时返回不可用，不影响批处理。"""
    g = gpu_state()
    if not g:
        return "GPU 温度不可用（未检测到 nvidia-smi）"
    return f"GPU {g[0]}°C / 利用率 {g[1]}% / 显存 {g[2]} MiB / 降频 {g[3]}"


def write_report(path: Path, cfg: dict, backend, limit: int, run_id: str,
                 stats: Counter, details: list[dict], metrics: dict, backup,
                 batch_size: int | None = None,
                 cooldown_seconds: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    lines = [
        f"# P2 单条分析引擎 · {limit} 条实测报告",
        "",
        f"- 生成时间（UTC）：{now}",
        f"- run_id：`{run_id}`",
        f"- 后端 / 模型：`{backend.name}` / `{backend.model}`",
        f"- prompt_version：`{cfg['prompt_version']}`",
        f"- 分批策略：每 {batch_size} 条为一批，批间冷却 {cooldown_seconds} 秒，并发 1 条"
        if batch_size else "- 分批策略：未分批（整批连续执行）",
        f"- 抽样范围：`{cfg.get('products', ['ChatGPT', 'Google Gemini', 'Perplexity'])}`，每产品均分名额、按星级均匀取样",
        f"- 写库前备份：`{backup}`" if backup else "- 写库前备份：无（库不存在）",
        "",
        "## 1. 核心指标",
        "",
        "| 指标 | 结果 | 说明 |",
        "|---|---|---|",
        f"| 实际调用模型 | {metrics['attempted']} 条 | 不含规则跳过与续跑跳过 |",
        f"| 结构化成功（done） | {metrics['done']} / {metrics['attempted']}（{metrics['success_rate']}） | schema 完整且引文命中 |",
        f"| 可用（done+degraded） | {metrics['done'] + metrics['degraded']} / {metrics['attempted']}（{metrics['usable_rate']}） | 含自动补齐，均需人工复核 |",
        f"| schema 完整率 | {metrics['schema_complete_rate']} | 无缺失、无非法取值 |",
        f"| 引文命中率 | {metrics['quote_hit_rate']} | 规范化空白后必须是原文连续子串 |",
        f"| degraded / failed / 规则跳过 | {metrics['degraded']} / {metrics['failed']} / {metrics['rule_skipped']} |  |",
        f"| 平均 / 中位 / 最长耗时 | {metrics['latency_avg_s']} / {metrics['latency_median_s']} / {metrics['latency_max_s']} s | 含 JSON 失败重试的两次调用 |",
        f"| 本次续跑跳过（已完成） | {metrics['resume_skipped']} | 断点续跑生效 |",
        f"| 达到重试上限跳过 | {metrics['max_attempts_skipped']} | 避免无限重试 |",
        "",
        "状态定义：`done` = 字段完整且引文命中；`degraded` = 已自动补齐缺失字段或引文未命中，必须人工复核；"
        "`failed` = 调用失败或两次 JSON 解析仍失败；`skipped` = 正文过短/无意义，规则直接跳过、不调用模型。",
        "",
        "## 2. 失败与降级分类",
        "",
        "| 类型 | 次数 |",
        "|---|---|",
    ]
    if metrics["failure_types"]:
        for k, v in sorted(metrics["failure_types"].items(), key=lambda x: -x[1]):
            lines.append(f"| `{k}` | {v} |")
    else:
        lines.append("| 无 | 0 |")
    if metrics["missing_fields"]:
        lines.append("")
        lines.append("缺失字段：")
        lines.append("")
        lines.append("| 字段 | 缺失次数 |")
        lines.append("|---|---|")
        for k, v in sorted(metrics["missing_fields"].items(), key=lambda x: -x[1]):
            lines.append(f"| `{k}` | {v} |")
    if metrics["flags"]:
        lines.append("")
        lines.append("规则打标：")
        lines.append("")
        lines.append("| 标记 | 次数 |")
        lines.append("|---|---|")
        for k, v in sorted(metrics["flags"].items(), key=lambda x: -x[1]):
            lines.append(f"| `{k}` | {v} |")

    lines += [
        "",
        "## 3. 逐条明细",
        "",
        "| # | 应用 | 星级 | 状态 | 耗时(s) | 尝试 | 引文命中 | 缺失字段 | 打标 | 原文摘要 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for i, d in enumerate(details, 1):
        lines.append(
            f"| {i} | {d.get('app')} | {d.get('score')}★ | {d['status']} | {d['seconds']} | "
            f"{d['attempts']} | {'是' if d.get('quote_ok') else '否'} | "
            f"{','.join(d.get('missing_fields') or []) or '-'} | "
            f"{','.join(x.split('(')[0] for x in (d.get('flags') or [])) or '-'} | "
            f"{_clip(d.get('content'), 60)} |"
        )

    bad = [d for d in details if d["status"] in {"failed", "degraded"}][:12]
    lines += ["", "## 4. 失败 / 降级案例（原文对照）", ""]
    if not bad:
        lines.append("无。")
    else:
        for i, d in enumerate(bad, 1):
            lines += [
                f"### {i}. feedback_id={d['feedback_id']} / {d.get('app')} / {d.get('score')}★ / {d['status']}",
                "",
                f"- 原文：{_clip(d.get('content'), 400)}",
                f"- 问题：{d['error']}",
                f"- 引文：{_clip(d.get('evidence_quote'), 120) or '（空）'}",
                f"- 原始输出：`{_clip(d.get('raw_output'), 600)}`",
                "",
            ]

    lines += [
        "",
        "## 5. 说明与局限",
        "",
        "- 本报告只覆盖本轮实际调用模型的样本，**不代表 450 条全量质量**；指标均标注实际样本量。",
        "- `degraded` 条目已自动补齐字段并强制 `needs_human_review=1`，不得直接作为结论使用。",
        "- 引文校验为规范化空白后的连续子串比对，不接受近义改写；未命中一律 `is_valid=0`。",
        "- 星级与情感/意图冲突只打标，不自动改写模型输出的情感判断。",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--report", default=None)
    ap.add_argument("--select-from", default=None,
                    help="按指定旧 prompt_version 的结果挑受影响评论（主题仅 other 或含 model_limits）")
    ap.add_argument("--select-remaining", default=None, metavar="NEW_VERSION",
                    help="补齐旧版 v2.0-p2 中尚无该 prompt_version 结果的评论")
    ap.add_argument("--select-theme", default=None, metavar="THEME",
                    help="选旧版结果中主题含指定主题、且新版尚无结果的评论")
    ap.add_argument("--from-version", default="v2.1-p2",
                    help="--select-theme 的源 prompt_version；默认 v2.1-p2")
    ap.add_argument("--feedback-ids", default=None,
                    help="逗号分隔的 feedback_id，定向重跑指定条目")
    ap.add_argument("--batch-size", type=int, default=60,
                    help="每批条数，批间冷却一次；默认 60")
    ap.add_argument("--cooldown-seconds", type=int, default=120,
                    help="批间冷却秒数；默认 120，设为 0 关闭冷却")
    ap.add_argument("--force", action="store_true",
                    help="忽略终态跳过，对同版本指定条目强制重跑（定向修复用）")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.model:
        cfg["backends"]["ollama"]["model"] = args.model
    limit = args.limit or int(cfg.get("batch", {}).get("trial_size", 30))

    backend = get_backend(cfg)
    if backend.name == "ollama":
        installed = list_ollama_models(cfg["backends"]["ollama"]["base_url"])
        if backend.model not in installed:
            print(json.dumps(
                {"error": "model_not_installed", "model": backend.model, "installed": installed},
                ensure_ascii=False, indent=2,
            ))
            return 2

    db_path = ROOT / "data" / "feedbacklens.db"
    backup = backup_if_exists(db_path)
    conn = init_db(db_path)
    if args.feedback_ids:
        ids = [int(x) for x in args.feedback_ids.split(",") if x.strip()]
        q = ",".join("?" * len(ids))
        rows = conn.execute(
            "SELECT f.id, f.review_id, f.content, f.score, p.name AS app "
            "FROM feedback f JOIN products p ON p.id=f.product_id "
            f"WHERE f.id IN ({q}) ORDER BY f.id",
            ids,
        ).fetchall()
        selected = [dict(r) for r in rows]
        limit = len(selected) or limit
    elif args.select_from:
        selected = select_affected(conn, backend.model, args.select_from)
        limit = len(selected) or limit
    elif args.select_theme:
        selected = select_theme(conn, backend.model, args.from_version,
                                args.select_theme, cfg["prompt_version"])
        limit = len(selected) or limit
    elif args.select_remaining:
        selected = select_remaining(conn, backend.model, "v2.0-p2", args.select_remaining)
        limit = len(selected) or limit
    else:
        selected = select_trial(conn, limit)
    if args.start or args.chunk:
        start = max(0, args.start)
        chunk = args.chunk or len(selected)
        items = selected[start:start + chunk]
    else:
        items = selected

    run_id = "p2-trial-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    batch_size = max(1, args.batch_size)
    cooldown = max(0, args.cooldown_seconds)
    batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)] or [[]]

    # 分批执行：并发恒为 1；每批结束后按需冷却，避免长时间连续推理过热
    stats: Counter = Counter()
    details: list[dict] = []
    for bi, batch in enumerate(batches, 1):
        if not batch:
            break
        if thermal_unsafe():
            print(f"[thermal] 开始前已达停止条件 · {gpu_thermal()}；停止推进，已完成结果已写库", flush=True)
            break
        print(f"[batch {bi}/{len(batches)}] {len(batch)} 条 · 开始前 {gpu_thermal()}", flush=True)
        s, d = run_batch(conn, batch, backend, cfg, run_id, force=args.force)
        stats.update(s)
        details.extend(d)
        calls = s.get("done", 0) + s.get("degraded", 0) + s.get("failed", 0)
        print(f"[batch {bi}/{len(batches)}] 完成 · 实际调用 {calls} 条 · {gpu_thermal()}", flush=True)
        if thermal_unsafe():
            print(f"[thermal] 达到停止条件 · {gpu_thermal()}；停止推进，已完成结果已写库", flush=True)
            break
        if bi < len(batches) and cooldown and calls:
            print(f"[cooldown] 冷却 {cooldown}s 降温中…", flush=True)
            time.sleep(cooldown)
            print(f"[cooldown] 冷却结束 · {gpu_thermal()}", flush=True)

    metrics = build_metrics(stats, details)
    slug = backend.model.replace(":", "_").replace(".", "_")
    report_path = Path(args.report) if args.report else (
        ROOT / "reports" / f"p2_analysis_trial_{limit}_{slug}.md"
    )
    write_report(report_path, cfg, backend, limit, run_id, stats, details, metrics, backup,
                 batch_size=batch_size, cooldown_seconds=cooldown)

    db_counts = {
        row["status"]: row["n"]
        for row in conn.execute("SELECT status, COUNT(*) AS n FROM analyses GROUP BY status")
    }
    conn.close()

    print(json.dumps(
        {
            "run_id": run_id,
            "model": backend.model,
            "prompt_version": cfg["prompt_version"],
            "requested": limit,
            "selected": len(items),
            "metrics": {k: v for k, v in metrics.items() if not isinstance(v, dict)},
            "failure_types": metrics["failure_types"],
            "db_status_counts": db_counts,
            "report": str(report_path),
        },
        ensure_ascii=False, indent=2,
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
