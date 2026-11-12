"""M6 优先级管理：加权评分、成本可改、人工改判留痕（谁改/原值/新值）。"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from app.ui import components, data, theme


def _auto_rationale(r: dict, w: dict) -> str:
    prod_txt = "、".join(f"{k} {v} 条" for k, v in sorted(r["products"].items())) or "未知"
    return (
        f"证据 {r['count']} 条（频率分 {r['freq_score']}/5，按 log 归一，不是简单按评论数排序）；"
        f"平均严重度 {r['severity_score']}/5；核心体验影响按 {r['impact_score']}/5 档；"
        f"影响范围 {r['scope_score']}/5（覆盖 {prod_txt}）；"
        f"实施成本 {r['cost_score']}/5（负向 −{w['cost']:.2f}）；"
        f"加权总分 {r['total_score']} → AI 分级 {r['ai_tier']}。"
        "该分级为规则计算，最终是否采纳需 PM 判断。"
    )


def _summary_table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{
        "排序": i,
        "问题": r["title"],
        "证据条数": r["count"],
        "频率分": r["freq_score"],
        "严重度": r["severity_score"],
        "体验影响": r["impact_score"],
        "影响范围": r["scope_score"],
        "成本": r["cost_score"],
        "总分": r["total_score"],
        "AI 分级": r["ai_tier"],
        "最终分级": r["pm_override_tier"] or r["ai_tier"],
        "判断理由": (r["rationale"] or "未保存（按默认参数计算）")[:80] + "…"
        if len(r["rationale"] or "") > 80 else (r["rationale"] or "未保存（按默认参数计算）"),
        "修改人": "—" if not r["saved"] else (r.get("pm_actor") or "PM"),
    } for i, r in enumerate(rows, 1)])


def render() -> None:
    conn = data.get_conn()
    components.page_header(
        "优先级管理",
        "按频率、严重度、核心体验影响、客户影响范围加权（成本为负向），给出 P1/P2/P3 与判断理由；"
        "权重与成本可调，人工改判全程留痕。",
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
        components.empty_state("当前版本没有足够证据（每个主题至少 3 条）形成候选问题。")
        return

    w = data.get_weights(conn, project_id)

    # ---------------- 权重设置
    st.markdown("**评分权重（成本为负向）**")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        wf = st.number_input("频率 0.30", 0.0, 1.0, float(w["freq"]), 0.05, format="%.2f")
    with c2:
        ws = st.number_input("严重度 0.25", 0.0, 1.0, float(w["severity"]), 0.05, format="%.2f")
    with c3:
        wi = st.number_input("核心体验 0.20", 0.0, 1.0, float(w["impact"]), 0.05, format="%.2f")
    with c4:
        wr = st.number_input("影响范围 0.15", 0.0, 1.0, float(w["scope"]), 0.05, format="%.2f")
    with c5:
        wc = st.number_input("成本 −0.10", 0.0, 1.0, float(w["cost"]), 0.05, format="%.2f")
    total_w = round(wf + ws + wi + wr + wc, 2)
    if abs(total_w - 1.0) > 0.001:
        components.note(f"当前权重之和为 {total_w}（建议为 1.00）。仍会按当前权重计算，"
                        "但不同条目之间的可比性会下降。")
    if st.button("保存权重并重算", type="primary"):
        data.save_weights(conn, project_id,
                          {"freq": wf, "severity": ws, "impact": wi, "scope": wr, "cost": wc})
        st.success("权重已保存，总分与分级已按新权重重算。")
        st.rerun()

    weights = {"freq": wf, "severity": ws, "impact": wi, "scope": wr, "cost": wc}
    # 用当前界面权重实时重算总分（保存前即可预览）
    for r in rows:
        r["total_score"] = data.score_total(r, weights)
        r["ai_tier"] = data.compute_tier(r["total_score"])
    rows.sort(key=lambda x: -x["total_score"])

    p_counts = {t: sum(1 for r in rows if (r["pm_override_tier"] or r["ai_tier"]) == t)
                for t in ("P1", "P2", "P3")}
    components.kpi_row([
        {"label": "候选问题", "value": len(rows), "hint": "仅纳入证据 ≥3 条的主题"},
        {"label": "P1", "value": p_counts["P1"], "hint": "总分 ≥4.0"},
        {"label": "P2", "value": p_counts["P2"], "hint": "总分 2.5–3.99"},
        {"label": "P3", "value": p_counts["P3"], "hint": "总分 <2.5"},
        {"label": "人工改判", "value": sum(1 for r in rows if r["pm_override_tier"]),
         "hint": "已保存的 PM 覆盖"},
    ])
    st.write("")
    st.markdown("**加权结果总表**")
    st.dataframe(_summary_table(rows), hide_index=True, width="stretch")
    st.caption("排序依据是加权总分，不是评论条数。频率分使用 log 归一，避免高样本产品单方面主导。")

    left, right = st.columns([1.1, 1])
    with left:
        fig = px.bar(pd.DataFrame([
            {"问题": r["title"], "总分": r["total_score"],
             "最终分级": r["pm_override_tier"] or r["ai_tier"]} for r in rows]),
            x="总分", y="问题", orientation="h", color="最终分级",
            color_discrete_map={"P1": theme.COLORS["danger"], "P2": theme.COLORS["warning"],
                                "P3": theme.COLORS["success"]},
            labels={"问题": ""})
        fig.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(theme.style_fig(fig, height=360, showlegend=True), width="stretch")
    with right:
        st.markdown("**分级规则**")
        st.markdown("- P1：加权总分 ≥ 4.0\n- P2：2.5 ≤ 总分 < 4.0\n- P3：总分 < 2.5")
        st.caption("总分 = 0.30×频率 + 0.25×严重度 + 0.20×体验影响 + 0.15×范围 − 成本×成本权重。"
                   "权重可调；成本、体验影响、范围可逐条修改；PM 可改判并留痕。")

    # ---------------- 逐条调整
    st.markdown("**逐条调整（成本 / 体验影响 / 范围 / 人工改判）**")
    for r in rows:
        cur_tier = r["pm_override_tier"] or r["ai_tier"]
        head = (f"{cur_tier} · {r['title']} · 总分 {r['total_score']} · "
                f"证据 {r['count']} 条")
        with st.expander(head):
            c1, c2, c3 = st.columns(3)
            with c1:
                cost = st.slider("实施成本（1 低 → 5 高）", 1, 5, int(r["cost_score"]),
                                 key=f"cost_{r['issue_id']}")
            with c2:
                impact = st.slider("核心体验影响（1 低 → 5 高）", 1, 5,
                                   int(round(r["impact_score"])),
                                   key=f"impact_{r['issue_id']}")
            with c3:
                scope = st.slider("客户影响范围（1 单产品 → 5 全产品/多数用户）", 1, 5,
                                  int(round(r["scope_score"])), key=f"scope_{r['issue_id']}")
            actor = st.text_input("修改人", value=st.session_state.get("pm_actor_default", "PM"),
                                  key=f"actor_{r['issue_id']}")
            tier_choice = st.selectbox(
                "人工改判最终分级", ["不覆盖", "P1", "P2", "P3"],
                index=(["不覆盖", "P1", "P2", "P3"].index(r["pm_override_tier"])
                       if r["pm_override_tier"] in ("P1", "P2", "P3") else 0),
                key=f"tier_{r['issue_id']}")
            reason = st.text_area(
                "判断理由（会写入审计）", value=r["rationale"] or _auto_rationale(r, weights),
                key=f"why_{r['issue_id']}", height=100)
            note = st.text_input("PM 备注（可选）", value=r["pm_note"] or "",
                                 key=f"note_{r['issue_id']}")

            preview_row = {**r, "cost_score": float(cost), "impact_score": float(impact),
                           "scope_score": float(scope)}
            preview = data.score_total(preview_row, weights)
            st.caption(f"按当前权重与本次调整，总分 {r['total_score']} → **{preview}**，"
                       f"规则分级 {data.compute_tier(preview)}；最终展示为 "
                       f"{tier_choice if tier_choice != '不覆盖' else data.compute_tier(preview)}。")
            if st.button("保存这一条", key=f"save_{r['issue_id']}"):
                preview_row["total_score"] = preview
                preview_row["ai_tier"] = data.compute_tier(preview)
                data.save_priority(conn, project_id, preview_row,
                                   None if tier_choice == "不覆盖" else tier_choice,
                                   note, reason, actor)
                st.session_state["pm_actor_default"] = actor.strip() or "PM"
                st.success(f"已保存：{r['title']}。原值与新值已写入审计记录。")
                st.rerun()

    components.divider()
    st.markdown("**修改审计（谁改 / 改成什么 / 原值）**")
    audit = data.list_priority_audit(conn, project_id)
    if audit:
        st.dataframe(pd.DataFrame([{
            "时间(UTC)": a["created_at"], "问题": a["title"] or f"issue #{a['issue_id']}",
            "修改人": a["actor"] or "-", "字段": a["field"],
            "原值": a["old_value"] if a["old_value"] is not None else "（无）",
            "新值": a["new_value"] if a["new_value"] is not None else "（无）",
        } for a in audit]), hide_index=True, width="stretch")
    else:
        st.caption("还没有保存记录。保存任一条目后，这里会显示原值、新值与修改人。")

    components.limitation_footer()
