"""M7 产品改进方案：痛点 → 建议（问题/证据/措施/预期改善/优先级/假设/不确定性）。

AI 生成的建议与 PM 已确认的决策在数据层（actions.status）与界面标识上都严格区分。
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app.ui import components, data, theme
from core.llm import get_backend, list_ollama_models, load_config

STATUS_LABELS = {
    "ai_draft": "AI 草案 · 未确认",
    "confirmed": "PM 已确认",
    "rejected": "PM 已驳回",
}


@st.cache_data(ttl=20, show_spinner=False)
def _models(base_url: str) -> list[str]:
    return list_ollama_models(base_url, timeout=4)


def _evidence_rows(df: pd.DataFrame, ids: list[int], limit: int = 6) -> pd.DataFrame:
    part = data.evidence_by_ids(df, ids)
    return part.head(limit)


def render() -> None:
    conn = data.get_conn()
    components.page_header(
        "产品改进方案",
        "把证据充分的痛点转成可执行建议；每条含问题、引文证据、措施、预期体验改善、优先级、"
        "假设与不确定性。AI 草案与 PM 确认的决策分开标识。",
    )
    project_id = st.session_state.get("project_id")
    if not project_id or not data.list_projects(conn):
        components.empty_state("请先在「项目工作区」创建或选择项目。")
        return
    versions = data.version_list(conn, project_id)
    if not versions:
        components.empty_state("该项目还没有分析结果，请先到「AI 分析」页运行。")
        return
    default = data.latest_version(conn, project_id)
    version = st.selectbox("分析版本", versions,
                           index=versions.index(default) if default in versions else 0)

    rows = data.priority_rows(conn, project_id, version)
    if not rows:
        components.empty_state("当前版本没有证据 ≥3 条的候选问题。")
        return
    df = data.load_insight_df(conn, project_id, version)
    acts = data.list_actions(conn, project_id)
    tier_of = {r["issue_id"]: (r["pm_override_tier"] or r["ai_tier"]) for r in rows}

    cfg = load_config()
    base_url = cfg.get("backends", {}).get("ollama", {}).get("base_url", "http://localhost:11434")
    models = _models(base_url)
    model = cfg.get("backends", {}).get("ollama", {}).get("model", "")
    model_ok = bool(models) and (model in models or any(m.split(":")[0] == model for m in models))

    n_draft = sum(1 for a in acts.values() if a["status"] == "ai_draft")
    n_ok = sum(1 for a in acts.values() if a["status"] == "confirmed")
    n_rej = sum(1 for a in acts.values() if a["status"] == "rejected")
    components.kpi_row([
        {"label": "候选问题", "value": len(rows), "hint": "证据 ≥3 条"},
        {"label": "AI 草案（未确认）", "value": n_draft, "hint": "模型生成，未采纳"},
        {"label": "PM 已确认", "value": n_ok, "hint": "人工决策，已留痕"},
        {"label": "PM 已驳回", "value": n_rej, "hint": "含驳回说明"},
        {"label": "尚未生成", "value": len(rows) - len(acts), "hint": "需先运行本地模型"},
    ])
    st.write("")
    st.markdown(
        components.pill("事实：证据与 feedback_id 可回溯", "success")
        + components.pill("AI 解释：模型建议，未经验证", "primary")
        + components.pill("假设：待验证", "warning")
        + components.pill("未知：证据不足", "danger"),
        unsafe_allow_html=True,
    )
    if not model_ok:
        components.note(f"未连上本地 Ollama（{base_url}）或未找到模型 {model}："
                        "可以查看已有方案与做人工确认，但暂不能生成新草案。")
    st.caption("生成使用本地模型（零付费），结果会记录模型与提示词版本；AI 草案不会被当成已确认决策。")

    st.markdown("**方案列表（AI 草案与人工决策分开显示）**")
    if acts:
        st.dataframe(pd.DataFrame([{
            "问题": a["title"], "优先级": a["priority_tier"] or tier_of.get(a["issue_id"], "-"),
            "状态": STATUS_LABELS.get(a["status"], a["status"]),
            "确认人": a["decided_by"] or "—", "模型": a["model_name"] or "—",
            "提示词版本": a["prompt_version"] or "—",
        } for a in acts.values()]), hide_index=True, width="stretch")

    for r in rows:
        tier = tier_of[r["issue_id"]]
        a = acts.get(r["issue_id"])
        state = STATUS_LABELS.get(a["status"], "尚未生成") if a else "尚未生成"
        head = f"{tier} · {r['title']} · {state} · 证据 {r['count']} 条"
        with st.expander(head, expanded=False):
            c1, c2 = st.columns([1.1, 1])
            with c1:
                st.markdown(f"**问题**：{r['title']}（主题 {data.theme_label(r['theme'])}）")
                st.markdown(
                    f"**优先级**：{tier}（总分 {r['total_score']}，AI 规则分级 {r['ai_tier']}）"
                    + (f"；PM 改判为 {r['pm_override_tier']}" if r["pm_override_tier"] else ""))
                st.markdown(f"**证据规模**：{r['count']} 条；涉及产品 "
                            + "、".join(f"{k} {v} 条" for k, v in sorted(r["products"].items()))
                            + "。")
                st.markdown("**引文证据（可回溯原文）**")
                components.evidence_table(_evidence_rows(df, r["ids"]),
                                          key=f"act_ev_{r['issue_id']}")
            with c2:
                if not a:
                    st.markdown("**建议措施**")
                    st.caption("尚未生成。点击下方按钮，用本地模型基于上述证据起草。")
                    if st.button("用本地模型生成改进草案", key=f"gen_{r['issue_id']}",
                                 disabled=not model_ok, type="primary"):
                        quotes = _evidence_rows(df, r["ids"], 8)["quote"].fillna("").tolist()
                        quotes = [q for q in quotes if q.strip()] or \
                                 _evidence_rows(df, r["ids"], 8)["content"].tolist()
                        with st.spinner(f"本地模型 {model} 正在起草…"):
                            try:
                                backend = get_backend(cfg)
                                draft = data.generate_action(backend, r, quotes)
                                data.save_action(conn, project_id, r["issue_id"], draft, tier)
                                st.success("草案已保存为 AI 草案（未确认），请人工复核后再确认。")
                                st.rerun()
                            except Exception as exc:  # 本地服务异常时给出明确原因
                                st.error(f"生成失败：{exc}")
                else:
                    st.markdown(components.badge("AI 建议（模型生成，未经验证）", "ai")
                                + components.badge(STATUS_LABELS.get(a["status"], a["status"]),
                                                   "fact" if a["status"] == "confirmed"
                                                   else "unknown"),
                                unsafe_allow_html=True)
                    st.markdown("**建议措施**")
                    st.write(a["proposal"] or "（模型未返回具体措施，需人工补充）")
                    st.markdown("**预期体验改善（AI 估计，未验证）**")
                    st.write(a["expected_effect"] or "—")
                    st.markdown("**假设**")
                    st.write(a["assumptions"] or "—")
                    st.markdown("**不确定性与风险**")
                    st.write(a["uncertainty"] or "—")
                    st.caption(f"来源：模型 {a['model_name'] or '—'} · "
                               f"提示词 {a['prompt_version'] or '—'} · "
                               f"状态 {STATUS_LABELS.get(a['status'], a['status'])}")

                    actor = st.text_input("决策人", value=st.session_state.get(
                        "pm_actor_default", "PM"), key=f"dec_actor_{r['issue_id']}")
                    note = st.text_input("决策说明 / 驳回理由", value=a["reject_reason"] or "",
                                         key=f"dec_note_{r['issue_id']}")
                    b1, b2, b3 = st.columns(3)
                    with b1:
                        if st.button("确认采纳", key=f"ok_{r['issue_id']}", type="primary"):
                            data.decide_action(conn, a["id"], "confirmed", actor, note)
                            st.session_state["pm_actor_default"] = actor.strip() or "PM"
                            st.success("已记录为 PM 确认的决策（人工确认，与 AI 草案分开）。")
                            st.rerun()
                    with b2:
                        if st.button("标记待验证", key=f"hold_{r['issue_id']}"):
                            data.decide_action(conn, a["id"], "ai_draft", actor,
                                               note or "待验证")
                            st.success("已标记为待验证，仍属 AI 草案。")
                            st.rerun()
                    with b3:
                        if st.button("驳回", key=f"no_{r['issue_id']}"):
                            data.decide_action(conn, a["id"], "rejected", actor,
                                               note or "未填写理由")
                            st.success("已驳回，理由已记录。")
                            st.rerun()
                    with st.expander("重新生成草案（会覆盖当前 AI 草案并重置确认状态）"):
                        if st.button("重新生成", key=f"regen_{r['issue_id']}",
                                     disabled=not model_ok):
                            quotes = _evidence_rows(df, r["ids"], 8)["quote"].fillna("").tolist()
                            quotes = [q for q in quotes if q.strip()]
                            try:
                                draft = data.generate_action(get_backend(cfg), r, quotes)
                                data.save_action(conn, project_id, r["issue_id"], draft, tier)
                                st.success("已重新生成 AI 草案（未确认）。")
                                st.rerun()
                            except Exception as exc:
                                st.error(f"生成失败：{exc}")

    components.limitation_footer()
