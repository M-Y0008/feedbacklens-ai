"""通用展示组件（只负责渲染，不查库、不算指标）。"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app.ui.theme import COLORS

BADGE_KINDS = {
    "fact": ("事实", "fl-badge-fact"),
    "ai": ("AI 解释", "fl-badge-ai"),
    "assume": ("假设", "fl-badge-assume"),
    "unknown": ("未知", "fl-badge-unknown"),
    "p1": ("P1", "fl-badge-p1"),
    "p2": ("P2", "fl-badge-p2"),
    "p3": ("P3", "fl-badge-p3"),
    "muted": ("", "fl-badge-muted"),
}


def badge(text: str, kind: str = "muted") -> str:
    _, cls = BADGE_KINDS.get(kind, ("", "fl-badge-muted"))
    return f'<span class="fl-badge {cls}">{text}</span>'


def sidebar_brand() -> None:
    st.markdown(
        '<div class="fl-brand"><div class="fl-brand-mark">FL</div>'
        '<div><div class="fl-brand-name">FeedbackLens AI</div>'
        '<div class="fl-brand-sub">用户之声洞察 · 本地运行</div></div></div>',
        unsafe_allow_html=True,
    )


def page_header(title: str, desc: str = "", extra_html: str = "") -> None:
    st.markdown(
        f'<div class="fl-page-head"><div class="fl-page-title">{title}</div>'
        f'<div class="fl-page-desc">{desc}</div>{extra_html}</div>',
        unsafe_allow_html=True,
    )


def section(title: str, sub: str = "") -> None:
    st.markdown(
        f'<div class="fl-section"><div class="fl-section-title">{title}</div>'
        f'<div class="fl-section-sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def kpi_row(items: list[dict]) -> None:
    """items: [{label, value, hint}]"""
    cols = st.columns(len(items))
    for col, it in zip(cols, items):
        with col:
            st.markdown(
                f'<div class="fl-kpi"><div class="fl-kpi-label">{it["label"]}</div>'
                f'<div class="fl-kpi-value">{it["value"]}</div>'
                f'<div class="fl-kpi-hint">{it.get("hint", "")}</div></div>',
                unsafe_allow_html=True,
            )


def note(text: str) -> None:
    st.markdown(f'<div class="fl-note">{text}</div>', unsafe_allow_html=True)
    st.write("")


def empty_state(text: str) -> None:
    st.markdown(f'<div class="fl-empty">{text}</div>', unsafe_allow_html=True)


def quote(text: str, meta: str = "") -> str:
    safe = (text or "").replace("<", "&lt;").replace(">", "&gt;")
    meta_html = f'<div class="fl-quote-meta">{meta}</div>' if meta else ""
    return f'<div class="fl-quote">{meta_html}{safe}</div>'


def evidence_table(df: pd.DataFrame, key: str | None = None) -> None:
    """证据表：每条都能看到 feedback_id 与原文，供下钻追溯。"""
    if df.empty:
        st.write("无匹配证据。")
        return
    show = df.rename(
        columns={
            "feedback_id": "feedback_id",
            "review_id": "review_id",
            "product": "产品",
            "score": "星级",
            "review_at": "评论时间(UTC)",
            "severity": "严重度",
            "sentiment": "情感",
            "needs_human_review": "需人工复核",
            "quote": "AI 引文",
            "content": "原文",
        }
    )
    st.dataframe(
        show,
        hide_index=True,
        width="stretch",
        column_config={
            "feedback_id": st.column_config.NumberColumn("feedback_id", width="small"),
            "星级": st.column_config.NumberColumn("星级", width="small"),
            "严重度": st.column_config.NumberColumn("严重度", width="small"),
            "需人工复核": st.column_config.CheckboxColumn("需人工复核", width="small"),
            "原文": st.column_config.TextColumn("原文", width="large"),
            "AI 引文": st.column_config.TextColumn("AI 引文", width="medium"),
        },
        key=key,
    )


def limitation_footer() -> None:
    st.markdown(
        '<div class="fl-divider"></div>'
        '<div style="color:#667085;font-size:.76rem;line-height:1.7">'
        "口径说明：所有数字来自本地 SQLite 中真实评论与模型输出；"
        "AI 解释与事实分开标识；样本量不足（证据 < 3 条）的条目仅作弱信号。"
        "相关性不等于因果。数据集地区、语言、时间窗有限，结论不可外推。"
        "</div>",
        unsafe_allow_html=True,
    )


def divider() -> None:
    st.markdown('<div class="fl-divider"></div>', unsafe_allow_html=True)


def badge_legend() -> None:
    st.markdown(
        badge("事实：可回溯原文", "fact")
        + badge("AI 解释：模型输出，需人工确认", "ai")
        + badge("假设：待验证", "assume")
        + badge("未知：证据不足", "unknown"),
        unsafe_allow_html=True,
    )


def pill(text: str, color: str = "primary") -> str:
    bg = {
        "primary": COLORS["primary_soft"],
        "success": COLORS["success_soft"],
        "warning": COLORS["warning_soft"],
        "danger": COLORS["danger_soft"],
    }.get(color, "#F2F4F7")
    fg = {
        "primary": COLORS["primary"],
        "success": COLORS["success"],
        "warning": COLORS["warning"],
        "danger": COLORS["danger"],
    }.get(color, COLORS["muted"])
    return f'<span class="fl-badge" style="background:{bg};color:{fg}">{text}</span>'
