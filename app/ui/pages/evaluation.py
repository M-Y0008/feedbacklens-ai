"""M9 AI 质量评估：人工标注（不预填）、AI/人工比对、错误分布、模型与提示词版本记录。"""
from __future__ import annotations

import json
from datetime import datetime

import pandas as pd
import streamlit as st

from app.ui import components, data

PER_PRODUCT = {10: "每产品 10 条（共 30）", 20: "每产品 20 条（共 60，正式标注）"}


def _err_flags(r: pd.Series) -> dict:
    gold = set(r["gold_themes"] or [])
    ai = set(r["themes"] or [])
    return {
        "主题未命中": not bool(gold & ai),
        "主题不完全一致": gold != ai,
        "情感不一致": r["gold_sentiment"] != r["sentiment"],
        "意图不一致": r["gold_intent"] != r["intent"],
        "严重度偏差>1": (pd.isna(r["gold_severity"]) or pd.isna(r["severity"])
                     or abs(float(r["gold_severity"]) - float(r["severity"])) > 1),
    }


def _error_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    flags = df.apply(_err_flags, axis=1, result_type="expand")
    base = pd.concat([df[["product", "gold_themes"]], flags], axis=1)
    by_prod = base.groupby("product").agg(
        n=("主题未命中", "size"), **{c: (c, "sum") for c in flags.columns}).reset_index()
    for c in flags.columns:
        by_prod[c + "率"] = (by_prod[c] / by_prod["n"] * 100).round(1).astype(str) + "%"
    rows = []
    for _, r in df.iterrows():
        g = r["gold_themes"] or ["（未标主题）"]
        for t in g:
            rows.append({"主题": data.theme_label(t), **_err_flags(r)})
    by_theme = pd.DataFrame(rows).groupby("主题").agg(
        n=("主题未命中", "size"), **{c: (c, "sum") for c in flags.columns}).reset_index()
    for c in flags.columns:
        by_theme[c + "率"] = (by_theme[c] / by_theme["n"] * 100).round(1).astype(str) + "%"
    return by_prod, by_theme


def render() -> None:
    conn = data.get_conn()
    components.page_header(
        "AI 质量评估",
        "人工标注 → 与 AI 结果逐字段比对 → 准确率与错误分布；标注数据只来自你本人的输入，"
        "系统不会预填或编造。",
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
    version = st.selectbox("对比分析版本", versions,
                           index=versions.index(default) if default in versions else 0)

    snap = data.model_config_snapshot()
    st.caption(f"模型 {snap['model']} · 提示词 {snap['prompt_version']} · "
               f"temperature {snap['temperature']} · num_ctx {snap['num_ctx']}")

    c1, c2 = st.columns([1, 2])
    with c1:
        per_product = st.selectbox("标注样本量", list(PER_PRODUCT),
                                   format_func=lambda k: PER_PRODUCT[k], index=1)
    with c2:
        labeler = st.text_input("标注人", value="PM")
    sample = data.sample_for_labeling(conn, project_id, version, per_product)
    if sample.empty:
        components.empty_state("当前版本没有可用于标注的样本。")
        return
    labels_all = data.list_labels(conn, project_id, version, labeler)
    sample_ids = set(sample["feedback_id"])
    # 只统计属于当前样本量设置的标注，避免 30 条与 60 条两套样本互相串数。
    labels = labels_all[labels_all["feedback_id"].isin(sample_ids)] \
        if not labels_all.empty else labels_all
    done_ids = set(labels["feedback_id"]) if not labels.empty else set()
    todo = sample[~sample["feedback_id"].isin(done_ids)]

    pct = round(len(done_ids) / len(sample) * 100, 1)
    components.kpi_row([
        {"label": "目标样本", "value": len(sample), "hint": PER_PRODUCT[per_product]},
        {"label": "已标注（当前标注人）", "value": len(done_ids), "hint": f"{pct}%"},
        {"label": "待标注", "value": len(todo), "hint": "不会自动填充"},
    ])
    st.progress(min(1.0, pct / 100), text=f"标注进度 {pct}%（{len(done_ids)} / {len(sample)}）")
    outside = len(labels_all) - len(labels) if not labels_all.empty else 0
    if outside:
        st.caption(f"另有 {outside} 条标注属于其他样本量设置，不计入当前 60/30 条统计。")

    tab1, tab2, tab3 = st.tabs(["人工标注", "AI vs 人工", "版本与历史"])

    with tab1:
        mode = st.radio("操作模式", ["标注未标条目", "修改已标注条目"],
                        horizontal=True, key="label_mode")
        st.markdown("**请先阅读原文再判断；AI 结果默认折叠，避免先入为主。**")
        pick, old = None, None
        if mode == "标注未标条目":
            if todo.empty:
                st.success("本批次已全部标注完成；可切到「修改已标注条目」回改，或到「AI vs 人工」看指标。")
            else:
                pick = st.selectbox(
                    "选择待标注评论", todo["feedback_id"].tolist(),
                    format_func=lambda i: (f"#{i} · "
                                           + str(todo.loc[todo["feedback_id"] == i, "product"].iloc[0])
                                           + " · "
                                           + str(todo.loc[todo["feedback_id"] == i, "content"].iloc[0])[:36]),
                )
                row = sample[sample["feedback_id"] == pick].iloc[0]
        elif labels.empty:
            st.info("当前样本量设置下还没有已标注条目，先切回「标注未标条目」。")
        else:
            pick = st.selectbox(
                "选择已标注条目（可改）", labels["feedback_id"].tolist(),
                format_func=lambda i: (f"#{i} · "
                                       + str(labels.loc[labels["feedback_id"] == i, "product"].iloc[0])
                                       + " · 已标 " +
                                       "、".join(data.theme_label(t) for t in
                                                labels.loc[labels["feedback_id"] == i,
                                                           "gold_themes"].iloc[0])),
            )
            row = sample[sample["feedback_id"] == pick].iloc[0]
            old = labels[labels["feedback_id"] == pick].iloc[0]

        if pick is not None:
            st.markdown(f"**原文**（#{row['feedback_id']} · {row['product']} · "
                        f"{row['review_at_str']} · 星级 {row['score']}）")
            st.write(row["content"])
            with st.expander("查看 AI 结果（参考，可能影响你的判断）"):
                st.write(f"主题：{'、'.join(data.theme_label(t) for t in (row['themes'] or [])) or '无'}")
                st.write(f"情感：{data.SENTIMENT_LABELS.get(row['sentiment'], row['sentiment'])}；"
                         f"意图：{data.INTENT_LABELS.get(row['intent'], row['intent'])}；"
                         f"严重度：{row['severity']}")
                st.caption(f"引文命中：{'是' if row['quote_ok'] else '否'}；"
                           f"需要人工复核：{'是' if row['needs_human_review'] else '否'}")
            if old is not None:
                st.caption("当前已存标注："
                           + "、".join(data.theme_label(t) for t in (old["gold_themes"] or []))
                           + f" · {data.SENTIMENT_LABELS.get(old['gold_sentiment'], old['gold_sentiment'])}"
                           f" · {data.INTENT_LABELS.get(old['gold_intent'], old['gold_intent'])}"
                           f" · 严重度 {old['gold_severity']} · 保存于 {old['labeled_at']}")

            with st.form(f"label_form_{pick}_{'edit' if old is not None else 'new'}"):
                themes = st.multiselect(
                    "人工标注主题（可多选）", list(data.THEME_LABELS),
                    format_func=lambda t: data.theme_label(t),
                    default=[t for t in (old["gold_themes"] or [])] if old is not None else [],
                    help="必须由你自行判断；若评论没有明确主题，选择「其他（证据不足）」。")
                sentiment = st.selectbox(
                    "人工标注情感", list(data.SENTIMENT_LABELS),
                    index=list(data.SENTIMENT_LABELS).index(old["gold_sentiment"])
                    if old is not None and old["gold_sentiment"] in data.SENTIMENT_LABELS else None,
                    placeholder="请选择情感",
                    format_func=lambda s: data.SENTIMENT_LABELS[s])
                intent = st.selectbox(
                    "人工标注意图", list(data.INTENT_LABELS),
                    index=list(data.INTENT_LABELS).index(old["gold_intent"])
                    if old is not None and old["gold_intent"] in data.INTENT_LABELS else None,
                    placeholder="请选择意图",
                    format_func=lambda s: data.INTENT_LABELS[s])
                severity = st.selectbox(
                    "人工标注严重度", list(range(6)),
                    index=int(old["gold_severity"]) if old is not None
                    and old["gold_severity"] is not None else None,
                    placeholder="请选择严重度（0 无问题 / 3 主流程受损 / 5 阻断）",
                    format_func=lambda s: data.SEVERITY_LABELS[s])
                notes = st.text_input("标注说明（可选）",
                                      value=(old["notes"] or "") if old is not None else "")
                submitted = st.form_submit_button(
                    "保存修改（覆盖并留痕）" if old is not None else "保存并下一条",
                    type="primary")
                if submitted:
                    if not themes or sentiment is None or intent is None or severity is None:
                        st.error("请先自行选择主题、情感、意图与严重度，系统不会替你填写。")
                    else:
                        overwritten = data.save_label(
                            conn, int(pick), labeler, themes, sentiment, intent, severity,
                            notes, per_product)
                        if overwritten:
                            st.success(f"已覆盖 #{pick} 的标注，原值与新值已写入修改记录。")
                        else:
                            st.success(f"已保存 #{pick} 的人工标注。")
                        st.rerun()
            st.caption("新建时四个字段都由你本人选择；修改模式会带回你此前的标注值。"
                       "覆盖写入会逐字段记录原值、新值与时间，不静默改数。")

    with tab2:
        if labels.empty:
            st.info("还没有人工标注。完成至少一条后，这里会显示 AI 与人工的比对指标。")
        else:
            m = data.compute_label_metrics(labels)
            components.kpi_row([
                {"label": "已标注", "value": m["labeled"], "hint": "当前标注人"},
                {"label": "主题命中率", "value": f"{m['theme_hit_rate']}%", "hint": "至少命中一个"},
                {"label": "主题完全一致", "value": f"{m['theme_exact_rate']}%", "hint": "集合相等"},
                {"label": "情感准确率", "value": f"{m['sentiment_acc']}%", "hint": ""},
                {"label": "意图准确率", "value": f"{m['intent_acc']}%", "hint": ""},
                {"label": "严重度±1", "value": f"{m['severity_within1']}%", "hint": "允许差 1 档"},
            ])
            st.write("")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**情感混淆矩阵（行=人工，列=AI）**")
                confusion = data.sentiment_confusion(labels)
                if confusion.empty:
                    st.caption("数据不足")
                else:
                    confusion.index = [data.SENTIMENT_LABELS.get(i, i) for i in confusion.index]
                    confusion.columns = [data.SENTIMENT_LABELS.get(c, c) for c in confusion.columns]
                    st.dataframe(confusion, width="stretch")
            with c2:
                st.markdown("**四字段全对比例**")
                st.metric("全字段完全一致", f"{m['all_fields_exact']}%",
                          help="主题集合、情感、意图、严重度（±1）全部一致")
                st.metric("引文命中率", f"{m['quote_hit_rate']}%",
                          help="AI 引文可在原文中找到的比例")
            by_prod, by_theme = _error_tables(labels)
            st.markdown("**按产品的错误分布**")
            st.dataframe(by_prod, hide_index=True, width="stretch")
            st.markdown("**按人工主题的错误分布**")
            st.dataframe(by_theme, hide_index=True, width="stretch")
            st.caption("错误率基于当前已标注样本，样本量小时波动大，不能作为最终结论。")
            if st.button("保存本次评估记录", type="primary"):
                run_id = f"eval-{datetime.now():%Y%m%d-%H%M%S}"
                data.save_eval_run(conn, run_id, len(labels), labeler, m, confusion,
                                   snap["model"], snap["prompt_version"])
                st.success(f"已保存评估记录 {run_id}（含模型与提示词版本）。")
                st.rerun()

    with tab3:
        st.markdown("**当前模型与提示词版本**")
        st.dataframe(pd.DataFrame([{
            "模型": snap["model"], "提示词版本": snap["prompt_version"],
            "temperature": snap["temperature"], "num_ctx": snap["num_ctx"],
            "后端": snap["active_backend"],
        }]), hide_index=True, width="stretch")
        st.markdown("**历史评估记录**")
        runs = data.list_eval_runs(conn)
        if not runs:
            st.caption("还没有保存的评估记录。完成标注后点击「保存本次评估记录」。")
        else:
            rows = []
            for r in runs:
                try:
                    metrics = json.loads(r["metrics_json"] or "{}")
                except json.JSONDecodeError:
                    metrics = {}
                rows.append({
                    "时间(UTC)": r["created_at"], "run_id": r["run_id"],
                    "样本量": r["sample_size"], "标注人": r["labeler"],
                    "模型": r["model_name"], "提示词": r["prompt_version"] or "-",
                    "情感准确率": metrics.get("sentiment_acc"),
                    "主题完全一致": metrics.get("theme_exact_rate"),
                    "全字段一致": metrics.get("all_fields_exact"),
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

        st.markdown("**标注修改痕迹（覆盖写入的原值 / 新值）**")
        audit = data.list_label_audit(conn, project_id, labeler)
        if audit.empty:
            st.caption("当前标注人还没有覆盖修改记录。修改已标注条目后，这里会按字段显示原值与新值。")
        else:
            audit["原文摘要"] = audit["content"].fillna("") + "…"
            st.dataframe(pd.DataFrame([{
                "时间(UTC)": r["changed_at"], "feedback_id": r["feedback_id"],
                "产品": r["product"], "字段": r["field"],
                "原值": r["old_value"] if r["old_value"] is not None else "（无）",
                "新值": r["new_value"] if r["new_value"] is not None else "（无）",
                "原文摘要": r["原文摘要"],
            } for _, r in audit.iterrows()]), hide_index=True, width="stretch")
            st.caption("记录表：eval_label_audit（按 feedback_id + 标注人 + 字段逐条记录，"
                       "只记录真正发生变化的字段）。")

    components.limitation_footer()
