"""M4 用户洞察 Dashboard：高频问题、痛点、功能需求、优势，逐条可下钻证据。"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.ui import components, data, theme


def _theme_chart(counts: dict, height: int = 340) -> go.Figure:
    items = sorted(counts.items(), key=lambda x: -x[1])[:9]
    labels = [data.theme_label(k) for k, _ in items]
    vals = [v for _, v in items]
    fig = px.bar(x=vals, y=labels, orientation="h",
                 labels={"x": "提及条数", "y": ""},
                 color_discrete_sequence=[theme.COLORS["primary"]])
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return theme.style_fig(fig, height=height)


def _cluster_list(df: pd.DataFrame, clusters: list[dict], key_prefix: str,
                  empty_text: str) -> None:
    if not clusters:
        components.empty_state(empty_text)
        return
    strong = [c for c in clusters if c["count"] >= 3]
    weak = [c for c in clusters if c["count"] < 3]
    if strong:
        st.markdown(f"**主要洞察（证据 ≥ 3 条）· {len(strong)} 项**")
        for i, c in enumerate(strong[:8], 1):
            prods = "、".join(c["products"])
            sev = f"平均严重度 {c['avg_severity']}" if c["avg_severity"] is not None else "严重度未标"
            with st.expander(f"{i}. {c['label']} · {c['count']} 条 · {prods} · {sev}"):
                ev = data.evidence_by_ids(df, c["ids"])
                components.evidence_table(ev, key=f"{key_prefix}_s{i}")
    if weak:
        with st.expander(f"弱信号（证据 < 3 条，仅提示，不作结论）· {len(weak)} 项"):
            st.dataframe(pd.DataFrame([
                {"主题": c["label"], "条数": c["count"], "产品": "、".join(c["products"]),
                 "feedback_id": "、".join(str(x) for x in c["ids"][:8])}
                for c in weak]), hide_index=True, width="stretch")


def render() -> None:
    conn = data.get_conn()
    components.page_header(
        "用户洞察 Dashboard",
        "从真实评论的 AI 结构化结果汇总；每条洞察都能下钻到 feedback_id 与原文证据。",
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
    c1, c2 = st.columns([1, 1])
    with c1:
        version = st.selectbox("分析版本", versions, index=versions.index(default)
                               if default in versions else 0)
    prods = [p["name"] for p in data.project_products(conn, project_id)]
    with c2:
        product = st.selectbox("产品筛选", ["全部产品"] + prods)

    stats = data.product_stats(conn, project_id, version)
    df = data.load_insight_df(conn, project_id, version,
                              None if product == "全部产品" else product)
    if df.empty:
        components.empty_state("当前筛选下没有分析结果。")
        return

    analyzed = df[df["status"].isin(["done", "degraded"])]
    tc = data.theme_counts(df)
    other_pct = tc["other_only"] / max(1, len(df)) * 100
    components.kpi_row([
        {"label": "样本量", "value": len(df), "hint": f"当前筛选：{product}"},
        {"label": "有效分析", "value": len(analyzed), "hint": "done + degraded"},
        {"label": "需人工复核", "value": int(df["needs_human_review"].sum()),
         "hint": "低置信 / 字段降级 / 星级冲突"},
        {"label": "仅 other（证据不足）", "value": f"{tc['other_only']}（{other_pct:.1f}%）",
         "hint": "不参与主题排行"},
    ])
    st.write("")
    badges = "".join(
        components.pill(f"{s['name']}：{s['samples']} 条 · "
                        f"{str(s['dmin'])[:10]}～{str(s['dmax'])[:10]}", "primary")
        for s in stats
    )
    st.markdown(badges, unsafe_allow_html=True)
    st.caption("各产品样本量与时间窗不同，占比差异不可直接视为优劣；样本不足的条目只作弱信号。")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["高频问题", "主要痛点", "功能需求", "产品优势"])
    with tab1:
        left, right = st.columns([1.15, 1])
        with left:
            st.markdown("**主题提及条数（不含 other 桶）**")
            st.plotly_chart(_theme_chart(tc["counts"]), width="stretch")
        with right:
            st.markdown("**主题排行**")
            rows = [{"主题": data.theme_label(k), "条数": v,
                     "占比": round(v / max(1, len(df)) * 100, 1)}
                    for k, v in sorted(tc["counts"].items(), key=lambda x: -x[1])]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            st.caption(f"注：other 出现 {tc['other_occ']} 次，"
                       f"仅 other 的评论 {tc['other_only']} 条，按规则不参与排行。")
        st.markdown("**按主题下钻证据**")
        for i, (code, n) in enumerate(sorted(tc["counts"].items(), key=lambda x: -x[1])[:8]):
            ids = df[df["themes"].map(lambda t: code in t)]["feedback_id"].tolist()
            with st.expander(f"{data.theme_label(code)}（{code}）· {n} 条证据"):
                components.evidence_table(
                    data.evidence_by_ids(df, [int(x) for x in ids]),
                    key=f"theme_{code}")

    with tab2:
        left, right = st.columns([1, 1.2])
        with left:
            st.markdown("**严重度分布（AI 判断，需人工复核）**")
            sev = data.severity_counts(df)
            if sev:
                fig = go.Figure(go.Bar(
                    x=[data.SEVERITY_LABELS[k] for k in sev],
                    y=list(sev.values()), marker_color=theme.COLORS["primary"]))
                fig.update_layout(xaxis_tickangle=-20)
                st.plotly_chart(theme.style_fig(fig, height=300), width="stretch")
                st.caption("严重度来自评论描述的具体问题，不由星级直接推断。")
        with right:
            st.markdown("**痛点聚类（pain_point 原文短语）**")
            clusters = data.pain_clusters(df)
            if clusters:
                st.dataframe(pd.DataFrame([
                    {"痛点": c["label"], "条数": c["count"],
                     "平均严重度": c["avg_severity"], "产品": "、".join(c["products"])}
                    for c in clusters[:10]]), hide_index=True, width="stretch")
            else:
                st.caption("当前样本没有被模型标注为具体痛点。")
        _cluster_list(df, clusters, "pain",
                      "没有可聚合的痛点原文（模型对泛泛评论按要求返回 null）。")

    with tab3:
        clusters = data.feature_clusters(df)
        left, right = st.columns([1, 1.1])
        with left:
            st.markdown("**功能需求 Top 10**")
            if clusters:
                top = clusters[:10]
                fig = px.bar(x=[c["count"] for c in top],
                             y=[c["label"][:48] for c in top], orientation="h",
                             labels={"x": "条数", "y": ""},
                             color_discrete_sequence=[theme.COLORS["accent"]])
                fig.update_layout(yaxis=dict(autorange="reversed"))
                st.plotly_chart(theme.style_fig(fig, height=340), width="stretch")
            else:
                st.caption("没有聚类后的功能需求。")
        with right:
            st.markdown("**需求明细与证据**")
            if clusters:
                st.dataframe(pd.DataFrame([
                    {"需求（原文短语）": c["label"], "条数": c["count"],
                     "产品": "、".join(c["products"])} for c in clusters[:10]]),
                    hide_index=True, width="stretch")
        _cluster_list(df, clusters, "feat", "没有模型识别出的明确功能请求。")

    with tab4:
        strengths = data.strengths(df)
        left, right = st.columns([1.3, 1])
        with left:
            st.markdown("**代表性正面原文**")
            if strengths:
                for s in strengths[:6]:
                    meta = (f"feedback_id {s['feedback_id']} · {s['product']} · "
                            f"{int(s['score']) if pd.notna(s['score']) else '-'}★ · "
                            f"主题 {', '.join(data.theme_label(t) for t in s['themes']) or '未归类'}")
                    st.markdown(components.quote(s["label"], meta), unsafe_allow_html=True)
            else:
                components.empty_state("当前筛选下没有正面/称赞类证据。")
        with right:
            st.markdown("**正面提及的主题（不含 other）**")
            pos = df[(df["sentiment"] == "positive") | (df["intent"] == "praise")]
            pc: dict[str, int] = {}
            for themes in pos["themes"]:
                for t in themes:
                    if t != "other":
                        pc[t] = pc.get(t, 0) + 1
            if pc:
                st.dataframe(pd.DataFrame([
                    {"主题": data.theme_label(k), "正面提及条数": v}
                    for k, v in sorted(pc.items(), key=lambda x: -x[1])]),
                    hide_index=True, width="stretch")
            else:
                st.caption("无正面主题提及。")

    components.limitation_footer()
