"""FeedbackLens AI — Streamlit 入口（P3 界面层）。

启动：streamlit run app/main.py --server.headless true --server.port 8501
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

st.set_page_config(
    page_title="FeedbackLens AI · 用户之声洞察",
    page_icon="FL",
    layout="wide",
    initial_sidebar_state="expanded",
)

from app.ui import components, data, theme  # noqa: E402
from app.ui.pages import (  # noqa: E402
    actions_page, analyze, compare, dashboard, evaluation, export, priorities, upload, workspace,
)

theme.inject()

PAGES = {
    "项目工作区": workspace.render,
    "CSV 上传与验证": upload.render,
    "AI 分析": analyze.render,
    "用户洞察 Dashboard": dashboard.render,
    "竞品对比": compare.render,
    "优先级管理": priorities.render,
    "产品改进方案": actions_page.render,
    "报告导出": export.render,
    "AI 质量评估": evaluation.render,
}
PAGE_SLUGS = {
    "workspace": "项目工作区",
    "upload": "CSV 上传与验证",
    "analyze": "AI 分析",
    "dashboard": "用户洞察 Dashboard",
    "compare": "竞品对比",
    "priorities": "优先级管理",
    "actions": "产品改进方案",
    "export": "报告导出",
    "evaluation": "AI 质量评估",
}


def _sidebar() -> str:
    with st.sidebar:
        components.sidebar_brand()
        st.markdown('<div class="fl-divider"></div>', unsafe_allow_html=True)

        projects = data.list_projects(data.get_conn())
        if projects:
            labels = {f"{p['name']}（{p['samples']} 条）": p["id"] for p in projects}
            current = st.session_state.get("project_id")
            keys = list(labels)
            default_idx = next((i for i, k in enumerate(keys) if labels[k] == current), 0)
            picked = st.selectbox("当前项目", keys, index=default_idx, key="sb_project")
            st.session_state["project_id"] = labels[picked]
        else:
            st.info("还没有项目，请先在「项目工作区」新建。")

        st.markdown('<div class="fl-divider"></div>', unsafe_allow_html=True)
        qp = st.query_params.get("page")
        if qp in PAGE_SLUGS and st.session_state.get("_qp_page") != qp:
            st.session_state["sb_page"] = PAGE_SLUGS[qp]
            st.session_state["_qp_page"] = qp
        page = st.radio("功能导航", list(PAGES.keys()), key="sb_page",
                        label_visibility="collapsed")
        st.markdown('<div class="fl-divider"></div>', unsafe_allow_html=True)
        st.caption("本地运行 · 零付费 API · 数据不出本机")
        st.caption("所有结论可回溯 feedback_id 与原文")
    return page


def main() -> None:
    page = _sidebar()
    PAGES[page]()


if __name__ == "__main__":
    main()
