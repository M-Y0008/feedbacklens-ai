"""M3 AI 分析：调用 core.batch 既有批处理，显示进度、状态分布与断点续跑。"""
from __future__ import annotations

import time
from collections import Counter
from datetime import datetime, timezone

import pandas as pd
import plotly.express as px
import streamlit as st

from app.ui import components, data, theme
from core.batch import run_batch
from core.llm import get_backend, list_ollama_models, load_config


@st.cache_data(ttl=20, show_spinner=False)
def _ollama_models(base_url: str) -> list[str]:
    return list_ollama_models(base_url, timeout=4)


def _run(cfg: dict, items: list[dict], version: str, force: bool, progress, status_box) -> Counter:
    conn = data.get_conn()
    backend = get_backend(cfg)
    run_id = f"ui-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    stats: Counter = Counter()
    consecutive_failures = 0
    fail_samples: list[str] = []
    for i, item in enumerate(items, 1):
        batch_stats, details = run_batch(conn, [item], backend, cfg, run_id, force=force)
        if batch_stats:
            stats.update(batch_stats)
        else:
            stats["resume_skipped"] += 1
        if details:
            d = details[0]
            if d["status"] == "failed":
                consecutive_failures += 1
                fail_samples.append(f"feedback_id={d['feedback_id']} · {d.get('error')}")
            else:
                consecutive_failures = 0
        progress.progress(i / max(1, len(items)),
                          text=f"{i}/{len(items)} · 本条 {details[0]['status'] if details else 'skipped'}")
        status_box.caption(
            f"实时计数：" + "，".join(f"{k} {v}" for k, v in stats.items())
            + f"｜run_id={run_id}")
        if consecutive_failures >= 3:
            st.error("连续 3 条失败，已停止本批（避免空转）。请检查 Ollama 服务或模型。")
            break
    if fail_samples:
        with st.expander("失败样本（最多 10 条）"):
            st.write("\n".join(fail_samples[:10]))
    return stats


def render() -> None:
    conn = data.get_conn()
    components.page_header(
        "AI 分析",
        "调用既有批处理逻辑（core.batch），逐条落库；同版本已完成的结果自动跳过，可断点续跑。",
    )
    project_id = st.session_state.get("project_id")
    if not project_id or not data.list_projects(conn):
        components.empty_state("请先在「项目工作区」创建或选择项目。")
        return

    cfg = load_config()
    versions = data.version_list(conn, project_id)
    default_v = cfg.get("prompt_version", "v2.2-p2")
    if default_v not in versions:
        versions = [default_v] + versions
    version = st.selectbox("分析版本（prompt_version）", versions,
                           index=versions.index(default_v))
    snapshot = data.model_config_snapshot()
    base_url = cfg.get("backends", {}).get("ollama", {}).get("base_url", "http://localhost:11434")
    models = _ollama_models(base_url)
    model_ok = snapshot["model"] in models or any(m.split(":")[0] == snapshot["model"] for m in models)
    st.markdown(
        components.pill(f"后端：{snapshot['active_backend']}", "primary")
        + components.pill(f"模型：{snapshot['model']}", "success" if model_ok else "danger")
        + components.pill(f"temperature={snapshot['temperature']} · num_ctx={snapshot['num_ctx']}",
                          "warning")
        + components.pill(f"prompt：{version}", "primary"),
        unsafe_allow_html=True,
    )
    if not models:
        components.note(f"未连上本地 Ollama（{base_url}）。请先启动 Ollama 服务；"
                        "已有分析结果仍可正常查看。")
    elif not model_ok:
        st.error(f"Ollama 已连接，但未找到配置模型 {snapshot['model']}；"
                 f"可用模型：{'、'.join(models[:10])}")

    counts = data.status_counts(conn, project_id, version)
    components.kpi_row([
        {"label": "样本量", "value": counts["samples"], "hint": "项目内已入库评论"},
        {"label": "已完成", "value": counts["done"], "hint": "status=done"},
        {"label": "降级 / 跳过", "value": f"{counts['degraded']} / {counts['skipped']}",
         "hint": "降级需人工复核"},
        {"label": "待分析", "value": counts["pending"], "hint": "同版本尚无结果"},
        {"label": "需人工复核", "value": counts["review_needed"], "hint": "含降级与低置信"},
    ])
    st.write("")

    summary = data.analysis_summary_rows(conn, project_id, version)
    if not summary.empty:
        st.markdown("**各产品状态分布**")
        st.dataframe(summary, hide_index=True, width="stretch")

    st.markdown("**执行分析**")
    mode = st.radio(
        "运行模式",
        ["续跑待分析", "重跑 failed / degraded（最多 20 条）", "重跑需人工复核（最多 20 条）"],
        horizontal=True,
    )
    pending_n = counts["pending"]
    run_disabled = (not models) or (not model_ok) or (pending_n == 0 and mode == "续跑待分析")
    if st.button("开始分析", type="primary", disabled=run_disabled):
        if mode == "续跑待分析":
            items = data.pending_items(conn, project_id, version)
            force = False
        elif mode.startswith("重跑 failed"):
            items = data.rerun_items(conn, project_id, version, ("failed", "degraded"), 20)
            force = True
        else:
            items = data.rerun_items(conn, project_id, version, ("degraded",), 20)
            force = True
        if not items:
            st.info("当前模式下没有需要处理的评论。")
        else:
            data.update_project_status(conn, project_id, "analyzing")
            progress = st.progress(0.0, text="准备中…")
            status_box = st.empty()
            t0 = time.perf_counter()
            stats = _run(cfg, items, version, force, progress, status_box)
            elapsed = time.perf_counter() - t0
            data.update_project_status(conn, project_id, "analyzed")
            st.success(f"本次处理 {len(items)} 条，耗时 {elapsed:.1f}s；结果："
                       + "，".join(f"{k} {v}" for k, v in stats.items()))
            st.session_state["last_analysis"] = {
                "version": version, "items": len(items), "elapsed": round(elapsed, 1),
                "stats": dict(stats),
            }
            st.rerun()

    if st.session_state.get("last_analysis"):
        last = st.session_state["last_analysis"]
        if last["version"] == version:
            st.info(f"最近一次界面运行：{last['items']} 条 / {last['elapsed']}s / "
                    f"计数 {last['stats']}")

    # 状态分布图（仅当前版本真实结果）
    rows = conn.execute(
        "SELECT a.status, COUNT(*) n FROM analyses a JOIN feedback f ON f.id=a.feedback_id "
        "WHERE f.project_id=? AND a.prompt_version=? AND a.model_name=? GROUP BY a.status",
        (project_id, version, snapshot["model"]),
    ).fetchall()
    if rows:
        st.markdown("**状态分布**")
        df = pd.DataFrame([dict(r) for r in rows])
        fig = px.bar(df, x="status", y="n", color="status",
                     color_discrete_map={"done": theme.COLORS["success"],
                                         "degraded": theme.COLORS["warning"],
                                         "skipped": theme.COLORS["muted"],
                                         "failed": theme.COLORS["danger"]},
                     labels={"status": "状态", "n": "条数"})
        st.plotly_chart(theme.style_fig(fig, height=230), width="stretch")
    components.limitation_footer()
