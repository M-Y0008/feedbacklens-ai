"""M2 CSV 上传与验证：字段识别、逐行校验、质量报告、确认后入库。"""
from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import streamlit as st

from app.ui import components, data, theme

FIELDS = [
    ("review_id", "评论 ID（必填）"),
    ("content", "评论正文（必填）"),
    ("score", "星级评分 1-5（必填）"),
    ("app_name", "应用/产品名（必填）"),
    ("at_utc", "评论时间（建议）"),
    ("package_name", "包名（可选）"),
]


def _quality_frames(val: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    q = val["quality"]
    rows = [{"指标": "总行数", "值": str(q["rows"])},
            {"指标": "通过校验", "值": str(q["accepted"])},
            {"指标": "被拒绝", "值": str(q["rejected"])},
            {"指标": "文件内重复 review_id", "值": str(q["duplicate_review_id"])},
            {"指标": "重复正文+应用", "值": str(q["duplicate_content"])},
            {"指标": "评分越界", "值": str(q["score_out_of_range"])},
            {"指标": "日期无法解析", "值": str(q["date_parse_fail"])},
            {"指标": "正文 <5 字符", "值": str(q["short_lt5"])},
            {"指标": "正文 <10 字符（分析会跳过）", "值": str(q["short_lt10"])},
            {"指标": "正文长度中位数", "值": str(q["content_len_median"])},
            {"指标": "含邮箱/链接", "值": str(q["sensitive_hits"])},
            {"指标": "时间范围(UTC)", "值": f"{q['date_min'][:10]} ～ {q['date_max'][:10]}"}]
    for reason, n in sorted(q["reason_counts"].items(), key=lambda x: -x[1]):
        rows.append({"指标": f"拒绝原因：{reason}", "值": str(n)})
    return pd.DataFrame(rows), pd.DataFrame(
        [{"应用": k, "行数": v} for k, v in q["app_counts"].items()])


def _preview_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    cols = [c for c in ["review_id", "app_name", "score", "review_at", "content", "reason", "flags"]
            if c in df.columns]
    return df[cols].head(20)


def render() -> None:
    conn = data.get_conn()
    components.page_header(
        "CSV 上传与验证",
        "自动识别字段 → 逐行校验 → 查看数据质量报告 → 确认后才写入 SQLite。",
    )
    project_id = st.session_state.get("project_id")
    projects = data.list_projects(conn)
    if not project_id or not projects:
        components.empty_state("请先在「项目工作区」创建或选择项目。")
        return
    project = data.project_by_id(conn, project_id)
    prod_names = [p["name"] for p in data.project_products(conn, project_id)]
    st.markdown(
        components.pill(f"当前项目：{project['name']}", "primary")
        + components.pill(f"项目产品：{'、'.join(prod_names) or '未设置'}", "warning"),
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader("上传评论 CSV（UTF-8 / GB18030 均可）", type=["csv"])
    if not uploaded:
        components.empty_state("等待上传 CSV。未点击「确认导入」前不会写入数据库。")
        _batches(conn, project_id)
        return

    file_key = f"{uploaded.name}:{uploaded.size}"
    if st.session_state.get("upload_key") != file_key:
        try:
            path = data.save_upload(uploaded)
            raw = data.read_uploaded_csv(path)
        except Exception as e:  # 上传解析失败必须可见，不静默
            st.error(f"CSV 解析失败：{type(e).__name__}: {e}")
            return
        st.session_state["upload_key"] = file_key
        st.session_state["upload_raw"] = raw
        st.session_state["upload_path"] = str(path)
        st.session_state.pop("last_import", None)

    raw: pd.DataFrame = st.session_state["upload_raw"]
    st.caption(f"已读取 {len(raw)} 行 × {len(raw.columns)} 列 · 文件：{uploaded.name}")

    auto = data.detect_mapping(list(raw.columns))
    with st.expander("字段映射（已自动识别，可手动修改）", expanded=not auto.get("content")):
        cols = ["（不映射）"] + [str(c) for c in raw.columns]
        mapping: dict[str, str] = {}
        mcols = st.columns(3)
        for i, (field, label) in enumerate(FIELDS):
            with mcols[i % 3]:
                picked = st.selectbox(
                    label, cols,
                    index=cols.index(auto[field]) if field in auto and auto[field] in cols else 0,
                    key=f"map_{field}_{file_key}",
                )
                if picked != "（不映射）":
                    mapping[field] = picked

    allow_new = st.checkbox("允许自动创建项目里不存在的产品", value=False)
    existing = {r[0] for r in conn.execute(
        "SELECT review_id FROM feedback WHERE project_id=?", (project_id,)).fetchall()}
    val = data.validate_upload(raw, mapping, prod_names, existing, allow_new)
    if not val.get("ok"):
        st.error(f"缺少必填字段映射：{'、'.join(val['missing_fields'])}")
        return

    q = val["quality"]
    components.kpi_row([
        {"label": "总行数", "value": q["rows"], "hint": "本次上传"},
        {"label": "可导入", "value": q["accepted"], "hint": "通过全部校验"},
        {"label": "被拒绝", "value": q["rejected"], "hint": "问题行见明细"},
        {"label": "时间范围", "value": f"{q['date_min'][:10]}", "hint": f"至 {q['date_max'][:10]} UTC"},
    ])
    st.write("")
    if q["unknown_apps"] and not allow_new:
        components.note(
            "以下应用不在项目产品列表中，相关行已被拒绝："
            + "、".join(q["unknown_apps"])
            + "。如确认要新增产品，可勾选上方“允许自动创建”。")

    left, right = st.columns([1.05, 1.0])
    with left:
        st.markdown("**质量报告**")
        metrics = pd.DataFrame([{"指标": r["指标"], "值": r["值"]}
                                 for r in _quality_frames(val)[0].to_dict("records")])
        st.dataframe(metrics, hide_index=True, width="stretch")
    with right:
        st.markdown("**评分分布**")
        sc = {int(k): v for k, v in q["score_counts"].items() if str(k).isdigit()}
        if sc:
            fig = px.bar(x=list(sc.keys()), y=list(sc.values()),
                         labels={"x": "星级", "y": "行数"},
                         color_discrete_sequence=[theme.COLORS["primary"]])
            st.plotly_chart(theme.style_fig(fig, height=200), width="stretch")
        st.markdown("**各应用行数**")
        ap = _quality_frames(val)[1]
        st.dataframe(ap, hide_index=True, width="stretch")

    st.markdown("**预览（前 20 行）**")
    tab_ok, tab_bad = st.tabs([f"可导入 {len(val['accepted'])}", f"被拒绝 {len(val['rejected'])}"])
    with tab_ok:
        st.dataframe(_preview_table(val["accepted"]), hide_index=True, width="stretch")
    with tab_bad:
        st.dataframe(_preview_table(val["rejected"]), hide_index=True, width="stretch")

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        st.download_button(
            "下载质量报告 CSV",
            metrics.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"quality_{uploaded.name}", mime="text/csv",
        )
    with c2:
        bad = val["rejected"]
        st.download_button(
            "下载被拒明细 CSV",
            (bad.to_csv(index=False).encode("utf-8-sig") if not bad.empty else b""),
            file_name=f"rejected_{uploaded.name}", mime="text/csv",
            disabled=bad.empty,
        )
    with c3:
        confirm = st.button("确认导入（写入数据库前自动备份）", type="primary",
                            disabled=len(val["accepted"]) == 0)

    if confirm:
        new_products = q["unknown_apps"] if allow_new else []
        res = data.commit_import(
            conn, project_id, val["accepted"], uploaded.name, q, new_products=new_products)
        st.session_state["last_import"] = res

    if st.session_state.get("last_import"):
        res = st.session_state["last_import"]
        st.success(
            f"导入完成：新增 {res['inserted']} 条，跳过已存在 {res['skipped_existing']} 条。"
            f"批次 ID：{res['batch_id']}；备份：{res['backup'] or '无（首次建库）'}")
        st.info("下一步：在「AI 分析」页触发分析，或到「用户洞察 Dashboard」查看已有分析结果。")

    _batches(conn, project_id)


def _batches(conn, project_id: int) -> None:
    batches = data.list_batches(conn, project_id)
    components.section("导入批次", "可按批次回滚；已产生分析结果的批次禁止回滚")
    if not batches:
        components.empty_state("暂无导入批次记录。")
        return
    rows = [{"批次 ID": b["id"][:8], "文件": b["filename"], "行数": b["row_count"],
             "入库": b["stored"], "拒绝": b["rejected_count"],
             "已分析": b["analyzed"], "时间(UTC)": b["created_at"][:19].replace("T", " ")}
            for b in batches]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    pick = st.selectbox("选择要回滚的批次", [b["id"] for b in batches],
                        format_func=lambda x: x[:8])
    sure = st.checkbox("我确认回滚该批次（删除该批次入库的评论）", key="rb_confirm")
    if st.button("回滚批次", disabled=not sure, key="rb_btn"):
        res = data.rollback_batch(conn, pick)
        if res["ok"]:
            st.success(f"已删除 {res['deleted']} 条并移除批次记录。备份：{res['backup']}")
            st.rerun()
        else:
            st.error(res["reason"])
