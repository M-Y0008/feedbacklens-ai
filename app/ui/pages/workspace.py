"""M1 项目工作区：创建项目、产品选择、状态与历史分析记录。"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app.ui import components, data, theme

DEFAULT_PRODUCTS = "ChatGPT_OpenAI, Google_Gemini, Perplexity_AI"


def _create_form(open_default: bool = False) -> None:
    with st.expander("新建项目", expanded=open_default):
        with st.form("create_project", clear_on_submit=True):
            name = st.text_input("项目名称", placeholder="例如：GenAI 三产品评论洞察（P4）")
            desc = st.text_area("项目描述", placeholder="想回答什么问题、数据从哪来、时间范围")
            products = st.text_input("参与对比的产品（逗号分隔）", value=DEFAULT_PRODUCTS)
            submitted = st.form_submit_button("创建项目", type="primary")
            if submitted:
                plist = [p.strip() for p in products.replace("，", ",").split(",") if p.strip()]
                if not name.strip():
                    st.error("项目名称必填。")
                elif len(plist) < 1:
                    st.error("至少选择一个产品。")
                else:
                    pid = data.create_project(data.get_conn(), name, desc, plist)
                    st.session_state["project_id"] = pid
                    st.success(f"已创建项目「{name.strip()}」，产品 {len(plist)} 个。")
                    st.rerun()


def _project_cards(projects: list[dict]) -> None:
    for i in range(0, len(projects), 3):
        cols = st.columns(3)
        for col, p in zip(cols, projects[i:i + 3]):
            with col:
                prods = data.project_products(data.get_conn(), p["id"])
                prod_names = "、".join(x["name"] for x in prods) or "未设置产品"
                st.markdown(
                    f'<div class="fl-card"><div style="display:flex;justify-content:space-between;'
                    f'align-items:center"><b>{p["name"]}</b>{components.badge(p["status"], "muted")}</div>'
                    f'<div style="color:#667085;font-size:.82rem;margin:.4rem 0 .6rem 0">'
                    f'{(p["description"] or "无描述")[:90]}</div>'
                    f'<div style="font-size:.8rem;color:#344054">样本量 <b>{p["samples"]}</b> · '
                    f'产品 {p["products"]} 个</div>'
                    f'<div style="font-size:.74rem;color:#98A2B3;margin-top:.35rem">{prod_names}</div>'
                    f'<div style="font-size:.72rem;color:#98A2B3;margin-top:.35rem">'
                    f'更新：{p["updated_at"][:19].replace("T", " ")} UTC</div></div>',
                    unsafe_allow_html=True,
                )
                if st.button("设为当前项目", key=f"pick_{p['id']}", width="stretch"):
                    st.session_state["project_id"] = p["id"]
                    st.rerun()


def _project_detail(project: dict) -> None:
    conn = data.get_conn()
    version = data.latest_version(conn, project["id"])
    summary = data.project_sample_summary(conn, project["id"])
    counts = data.status_counts(conn, project["id"], version)
    components.section(f"当前项目 · {project['name']}", f"数据按 project_id={project['id']} 隔离")
    components.kpi_row([
        {"label": "样本量", "value": summary["samples"], "hint": "已入库评论"},
        {"label": "参与产品", "value": len(data.project_products(conn, project["id"])),
         "hint": "按产品分组分析"},
        {"label": "当前分析版本", "value": version, "hint": f"已完成 {counts['done']} 条"},
        {"label": "需人工复核", "value": counts["review_needed"], "hint": "needs_human_review=1"},
    ])
    st.write("")
    left, right = st.columns([1.1, 1.9])
    with left:
        st.markdown("**状态与时间窗**")
        st.write(f"项目状态：`{project['status']}`")
        st.write(f"评论时间范围：{str(summary['dmin'] or '-')[:10]} ～ {str(summary['dmax'] or '-')[:10]} (UTC)")
        st.write(f"创建时间：{project['created_at'][:19].replace('T', ' ')} UTC")
        new_status = st.selectbox(
            "更新项目状态", ["draft", "analyzing", "analyzed", "archived"],
            index=["draft", "analyzing", "analyzed", "archived"].index(project["status"])
            if project["status"] in ("draft", "analyzing", "analyzed", "archived") else 0,
            key=f"status_{project['id']}",
        )
        if new_status != project["status"] and st.button("保存状态", key=f"save_status_{project['id']}"):
            data.update_project_status(conn, project["id"], new_status)
            st.success("状态已更新。")
            st.rerun()
    with right:
        st.markdown("**各产品样本量与时间范围**")
        rows = []
        for p in data.project_products(conn, project["id"]):
            stats = conn.execute(
                "SELECT COUNT(*) n, MIN(review_at) dmin, MAX(review_at) dmax "
                "FROM feedback WHERE project_id=? AND product_id=?",
                (project["id"], p["id"]),
            ).fetchone()
            rows.append({
                "产品": p["name"], "样本量": stats["n"],
                "时间范围": f"{str(stats['dmin'])[:10]} ～ {str(stats['dmax'])[:10]}",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    components.section("历史分析记录", "每次运行按 run_id / 模型 / prompt 版本留痕")
    hist = pd.DataFrame(data.run_history(conn, project["id"]))
    if hist.empty:
        components.empty_state("该项目还没有分析记录。")
    else:
        hist = hist.rename(columns={
            "run_id": "run_id", "model_name": "模型", "prompt_version": "prompt 版本",
            "started_at": "开始(UTC)", "finished_at": "结束(UTC)", "rows": "条数",
            "done": "done", "degraded": "degraded", "skipped": "skipped", "failed": "failed",
        })
        st.dataframe(hist, hide_index=True, width="stretch")

    archived = [p for p in data.list_projects(conn, include_deleted=True) if p["deleted_at"]]
    if archived:
        with st.expander("回收站（软删除项目）"):
            for p in archived:
                c1, c2 = st.columns([3, 1])
                c1.write(f"{p['name']} · 删除于 {p['deleted_at'][:19]}")
                if c2.button("恢复", key=f"restore_{p['id']}"):
                    data.restore_project(conn, p["id"])
                    st.rerun()

    with st.expander("危险操作：软删除当前项目"):
        st.caption("软删除不删除评论与分析数据，可在回收站恢复；按约定需二次确认。")
        sure = st.checkbox("我确认要移除该项目", key=f"del_ck_{project['id']}")
        if st.button("软删除项目", disabled=not sure, key=f"del_{project['id']}"):
            data.soft_delete_project(conn, project["id"])
            st.session_state.pop("project_id", None)
            st.success("已移入回收站。")
            st.rerun()


def render() -> None:
    conn = data.get_conn()
    projects = data.list_projects(conn)
    components.page_header(
        "项目工作区",
        "创建项目、维护参与对比的产品，查看样本状态与每次 AI 分析的历史记录。",
    )
    components.badge_legend()
    st.write("")

    if not projects:
        components.empty_state("还没有项目。先在下面创建第一个项目。")
    else:
        _project_cards(projects)

    _create_form(open_default=not projects)

    current_id = st.session_state.get("project_id")
    if current_id:
        project = data.project_by_id(conn, current_id)
        if project:
            components.divider()
            _project_detail(project)
    components.limitation_footer()
