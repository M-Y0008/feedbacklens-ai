"""M5 竞品对比：三产品在主题、情感、严重度上的差异，并显式标注样本覆盖与局限。"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.ui import components, data, theme


def _coverage_table(rows: list[dict]) -> pd.DataFrame:
    out = []
    for r in rows:
        out.append({
            "产品": r["product"],
            "样本量": r["samples"],
            "评论时间范围": f"{r['dmin'].strftime('%Y-%m-%d')} ~ {r['dmax'].strftime('%Y-%m-%d')}"
            if pd.notna(r["dmin"]) else "-",
            "覆盖天数": r["days"],
            "仅 other（证据不足）": r["other_only"],
            "引文命中率": f"{r['quote_hit']}%",
        })
    return pd.DataFrame(out)


def _theme_long(rows: list[dict], top_n: int = 8) -> pd.DataFrame:
    total: dict[str, int] = {}
    for r in rows:
        for k, v in r["theme_counts"].items():
            total[k] = total.get(k, 0) + v
    keep = [k for k, _ in sorted(total.items(), key=lambda x: -x[1])[:top_n]]
    recs = []
    for r in rows:
        for k in keep:
            n = r["theme_counts"].get(k, 0)
            recs.append({"产品": r["product"], "主题": data.theme_label(k),
                         "提及占比(%)": round(n / max(1, r["samples"]) * 100, 1),
                         "条数": n})
    return pd.DataFrame(recs)


def _severity_long(rows: list[dict], df: pd.DataFrame) -> pd.DataFrame:
    recs = []
    for r in rows:
        g = df[df["product"] == r["product"]]
        counts = data.severity_counts(g)
        for k in range(6):
            recs.append({"产品": r["product"], "严重度": data.SEVERITY_LABELS[k],
                         "条数": counts.get(k, 0)})
    return pd.DataFrame(recs)


def render() -> None:
    conn = data.get_conn()
    components.page_header(
        "竞品对比",
        "按产品比较主题、情感与严重度差异；同时在页面顶部展示各产品真实样本量与评论时间窗。",
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
    df = data.load_insight_df(conn, project_id, version)
    if df.empty:
        components.empty_state("当前版本没有分析结果。")
        return

    rows = data.product_compare_rows(df)
    components.kpi_row([
        {"label": "产品数", "value": len(rows), "hint": "全部来自同一分析版本"},
        {"label": "样本总量", "value": len(df), "hint": f"版本 {version}"},
        {"label": "本窗口跨度", "value": f"{df['review_at'].min():%m-%d} ~ "
                                        f"{df['review_at'].max():%m-%d}",
         "hint": "各产品窗口不同，见下表"},
        {"label": "仅 other（证据不足）", "value": int(sum(r["other_only"] for r in rows)),
         "hint": "不参与主题排行"},
    ])
    st.write("")

    # 样本覆盖：必须先看这里，避免误读为长期趋势
    st.markdown("**样本覆盖（先读这一行再看差异）**")
    cover = _coverage_table(rows)
    st.dataframe(cover, hide_index=True, width="stretch")
    windows = "；".join(
        f"{r['product']} 约 {r['days']} 天" + ("（约 1 个月）" if r["days"] >= 20 else "")
        for r in rows
    )
    components.note(
        f"各产品评论时间窗不同：{windows}。下面的主题/情感/严重度差异只描述各自窗口内的样本，"
        "样本量均为 "
        + "、".join(f"{r['samples']} 条" for r in rows)
        + "，不能解读为长期趋势、市场份额或产品优劣定论。"
    )

    tab1, tab2, tab3 = st.tabs(["主题差异", "情感与严重度", "证据下钻"])

    with tab1:
        left, right = st.columns([1.25, 1])
        with left:
            st.markdown("**主题提及占比（按各产品自身样本归一，不含 other）**")
            long = _theme_long(rows)
            fig = px.bar(long, x="提及占比(%)", y="主题", color="产品",
                         orientation="h", barmode="group",
                         color_discrete_map={r["product"]: theme.product_color(r["product"], i)
                                             for i, r in enumerate(rows)},
                         labels={"主题": ""})
            fig.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(theme.style_fig(fig, height=420, showlegend=True),
                            width="stretch")
            st.caption("占比 = 该主题被提及条数 / 该产品样本量；一条评论可命中多个主题。")
        with right:
            st.markdown("**主题 × 产品提及条数**")
            codes = sorted({k for r in rows for k in r["theme_counts"]},
                           key=lambda c: -sum(r["theme_counts"].get(c, 0) for r in rows))
            table = []
            for c in codes[:10]:
                rec = {"主题": data.theme_label(c)}
                for r in rows:
                    rec[r["product"]] = r["theme_counts"].get(c, 0)
                table.append(rec)
            st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")

        st.markdown("**各产品本窗口内的主要主题（事实性摘录，不代表长期结论）**")
        for r in rows:
            top = "、".join(f"{t}（{n} 条）" for t, n in r["top_themes"][:3]) or "无"
            st.markdown(f"- **{r['product']}**：样本 {r['samples']} 条，"
                        f"时间窗 {r['dmin']:%Y-%m-%d} ~ {r['dmax']:%Y-%m-%d}（约 {r['days']} 天），"
                        f"提及最多：{top}。")

    with tab2:
        left, right = st.columns([1, 1])
        with left:
            st.markdown("**情感分布（占各产品样本比例）**")
            sent = []
            for r in rows:
                for k, v in r["sentiment"].items():
                    sent.append({"产品": r["product"], "情感": data.SENTIMENT_LABELS.get(k, k),
                                 "占比(%)": round(v / max(1, r["samples"]) * 100, 1)})
            fig = px.bar(pd.DataFrame(sent), x="产品", y="占比(%)", color="情感",
                         barmode="stack",
                         color_discrete_map={"正面": theme.COLORS["success"],
                                             "中性": theme.COLORS["muted"],
                                             "褒贬混合": theme.COLORS["warning"],
                                             "负面": theme.COLORS["danger"]})
            st.plotly_chart(theme.style_fig(fig, height=320, showlegend=True),
                            width="stretch")
        with right:
            st.markdown("**严重度分布（条数）**")
            sev = _severity_long(rows, df)
            fig = px.bar(sev, x="严重度", y="条数", color="产品", barmode="group",
                         color_discrete_map={r["product"]: theme.product_color(r["product"], i)
                                             for i, r in enumerate(rows)})
            fig.update_layout(xaxis_tickangle=-25)
            st.plotly_chart(theme.style_fig(fig, height=320, showlegend=True),
                            width="stretch")

        st.markdown("**严重度指标对比**")
        st.dataframe(pd.DataFrame([{
            "产品": r["product"], "样本量": r["samples"],
            "平均严重度": r["avg_severity"],
            "严重度≥4 条数": r["sev4_plus"],
            "严重度≥4 占比": f"{round(r['sev4_plus'] / max(1, r['samples']) * 100, 1)}%",
            "负面占比": f"{r['sentiment_pct'].get('negative', 0)}%",
            "正面占比": f"{r['sentiment_pct'].get('positive', 0)}%",
        } for r in rows]), hide_index=True, width="stretch")
        st.caption("严重度由模型依据评论描述判断，不由星级直接推断；跨产品差异受样本窗口与抽样限制。")

    with tab3:
        left, right = st.columns([1, 1])
        prods = [r["product"] for r in rows]
        with left:
            picked = st.selectbox("产品", prods, key="cmp_prod")
        with right:
            g = df[df["product"] == picked]
            tc = data.theme_counts(g)["counts"]
            codes = sorted(tc, key=lambda c: -tc[c])
            theme_pick = st.selectbox("主题", ["全部主题"] + codes,
                                     format_func=lambda c: "全部主题" if c == "全部主题"
                                     else data.theme_label(c), key="cmp_theme")
        part = df[df["product"] == picked]
        if theme_pick != "全部主题":
            part = part[part["themes"].map(lambda t: theme_pick in t)]
        st.caption(f"{picked} · {theme_pick} · 共 {len(part)} 条证据，按严重度与星级排序。")
        components.evidence_table(
            data.evidence_by_ids(part, [int(x) for x in part["feedback_id"].tolist()]),
            key="cmp_evidence")

    components.limitation_footer()
