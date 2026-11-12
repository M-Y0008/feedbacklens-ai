"""M8 报告导出：Excel 九表真实生成与下载；PDF 无可靠免费方案时给出 HTML 打印替代。"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app.ui import components, data

EXCEL_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
SHEET_DESC = [
    ("项目概述", "项目、模型、提示词版本、样本总量与口径"),
    ("数据来源与样本覆盖", "各产品样本量、评论起止、覆盖天数"),
    ("数据质量", "分析状态、人工复核、引文命中率、other 桶"),
    ("核心洞察", "主题提及、占比、平均严重度、涉及产品"),
    ("痛点", "痛点原文短语、条数、严重度、feedback_id 证据"),
    ("竞品对比", "三产品主题/情感/严重度与时间窗备注"),
    ("优先级", "AI 分级、最终分级、分项得分、判断理由"),
    ("改进建议", "AI 草案与 PM 确认状态、假设、不确定性、模型版本"),
    ("局限说明", "样本窗口、抽样、AI 解释边界与不可外推事项"),
]


def render() -> None:
    conn = data.get_conn()
    components.page_header(
        "报告导出",
        "Excel 为必做项，包含项目概述、样本覆盖、数据质量、核心洞察、痛点、竞品对比、"
        "优先级、改进建议与局限说明；所有内容来自本地真实数据。",
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
    version = st.selectbox("导出分析版本", versions,
                           index=versions.index(default) if default in versions else 0)

    exports = data.list_exports(conn, project_id)
    components.kpi_row([
        {"label": "工作表", "value": f"{len(SHEET_DESC)} 张", "hint": "Excel 固定结构"},
        {"label": "历史导出", "value": len(exports), "hint": "记录在 export_logs"},
        {"label": "当前版本", "value": version, "hint": "内容随版本变化"},
    ])
    st.write("")

    st.markdown("**Excel 工作表结构**")
    st.dataframe(pd.DataFrame([{"工作表": n, "内容": d} for n, d in SHEET_DESC]),
                 hide_index=True, width="stretch")

    components.divider()
    st.markdown("**Excel（必做）**")
    if st.button("生成 Excel 并保存到本地", type="primary"):
        with st.spinner("正在汇总真实数据并生成 .xlsx…"):
            info = data.build_excel(conn, project_id, version)
        st.session_state["xlsx_export"] = {**info, "version": version}
        st.success(f"已生成：{info['filename']}（{info['rows']} 行数据）")
        st.caption(f"本地路径：{info['path']}")

    exp = st.session_state.get("xlsx_export")
    if exp and exp.get("version") == version:
        st.download_button("下载 Excel 报告", data=exp["bytes"], file_name=exp["filename"],
                           mime=EXCEL_MIME, type="primary")
        st.caption("下载的是刚生成的真实 .xlsx 文件，不是占位文件。")

    components.divider()
    st.markdown("**PDF 说明**")
    components.note(
        "当前没有启用直接导出 PDF：本项目只使用零付费、本地可运行的依赖，"
        "未安装 PDF 生成引擎，也不会用付费库伪装支持。"
        "替代方案：生成自包含的可打印 HTML，用浏览器打开后「打印 → 另存为 PDF」，"
        "内容与 Excel 一致，完全本地、免费。"
    )
    if st.button("生成可打印 HTML（PDF 替代）"):
        with st.spinner("正在生成可打印报告…"):
            info = data.build_html(conn, project_id, version)
        st.session_state["html_export"] = {**info, "version": version}
        st.success(f"已生成：{info['filename']}（{info['rows']} 行数据）")
        st.caption(f"本地路径：{info['path']}；打开后使用浏览器打印即可另存为 PDF。")

    exp2 = st.session_state.get("html_export")
    if exp2 and exp2.get("version") == version:
        st.download_button("下载可打印 HTML", data=exp2["bytes"], file_name=exp2["filename"],
                           mime="text/html")
        st.caption("提示：浏览器中 Ctrl+P → 目标选择「另存为 PDF」。")

    components.divider()
    st.markdown("**导出记录**")
    if exports:
        st.dataframe(pd.DataFrame([{
            "时间(UTC)": e["created_at"], "格式": e["format"].upper(),
            "文件名": e["filename"], "行数": e["rows"], "路径": e["path"],
        } for e in exports]), hide_index=True, width="stretch")
    else:
        st.caption("还没有导出记录。生成一次后会写入 export_logs 表。")

    components.limitation_footer()
