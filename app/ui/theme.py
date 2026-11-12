"""界面主题：统一设计语言（B2B SaaS 风格），只做展示，不含业务逻辑。"""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

# 设计令牌（与 .streamlit/config.toml 保持一致）
COLORS = {
    "bg": "#F6F8FB",
    "surface": "#FFFFFF",
    "border": "#E3E8EF",
    "text": "#101828",
    "muted": "#667085",
    "primary": "#0F5C8C",
    "primary_soft": "#E8F1F7",
    "accent": "#0E9384",
    "warning": "#B54708",
    "warning_soft": "#FEF0C7",
    "danger": "#B42318",
    "danger_soft": "#FEE4E2",
    "success": "#027A48",
    "success_soft": "#D1FADF",
    "violet": "#6941C6",
    "line": "#98A2B3",
}

PRODUCT_COLORS = {
    "ChatGPT_OpenAI": "#0F5C8C",
    "Google_Gemini": "#0E9384",
    "Perplexity_AI": "#6941C6",
}
FALLBACK_PALETTE = ["#0F5C8C", "#0E9384", "#6941C6", "#B54708", "#B42318", "#475467"]

CSS = f"""
<style>
:root {{
  --fl-bg: {COLORS['bg']};
  --fl-surface: {COLORS['surface']};
  --fl-border: {COLORS['border']};
  --fl-text: {COLORS['text']};
  --fl-muted: {COLORS['muted']};
  --fl-primary: {COLORS['primary']};
}}
html, body, [data-testid="stAppViewContainer"] {{
  background: var(--fl-bg);
  color: var(--fl-text);
  font-family: "Inter", "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
}}
[data-testid="stHeader"] {{ background: rgba(246,248,251,0.85); }}
[data-testid="stSidebar"] {{
  background: #FFFFFF;
  border-right: 1px solid var(--fl-border);
}}
[data-testid="stSidebar"] .block-container {{ padding-top: 1.2rem; }}
.block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1320px; }}

h1, h2, h3, h4 {{ letter-spacing: -0.01em; color: var(--fl-text); }}
h1 {{ font-size: 1.75rem; font-weight: 700; }}
h2 {{ font-size: 1.25rem; font-weight: 650; margin-top: 1.4rem; }}
h3 {{ font-size: 1.02rem; font-weight: 600; }}

.fl-brand {{ display:flex; align-items:center; gap:10px; margin-bottom: 6px; }}
.fl-brand-mark {{
  width: 34px; height: 34px; border-radius: 9px;
  background: linear-gradient(135deg, {COLORS['primary']}, {COLORS['accent']});
  color:#fff; display:flex; align-items:center; justify-content:center;
  font-weight: 700; font-size: 15px; letter-spacing: .5px;
}}
.fl-brand-name {{ font-weight: 700; font-size: 1.05rem; color: var(--fl-text); }}
.fl-brand-sub {{ font-size: .78rem; color: var(--fl-muted); margin-top: -2px; }}

.fl-page-head {{ margin: 0 0 1.1rem 0; }}
.fl-page-title {{ font-size: 1.65rem; font-weight: 700; margin: 0; }}
.fl-page-desc {{ color: var(--fl-muted); font-size: .92rem; margin-top: .35rem; }}

.fl-card {{
  background: var(--fl-surface);
  border: 1px solid var(--fl-border);
  border-radius: 12px;
  padding: 16px 18px;
  box-shadow: 0 1px 2px rgba(16,24,40,0.04);
  margin-bottom: 12px;
}}
.fl-kpi {{
  background: var(--fl-surface); border: 1px solid var(--fl-border);
  border-radius: 12px; padding: 14px 16px 12px 16px; height: 100%;
  box-shadow: 0 1px 2px rgba(16,24,40,0.04);
}}
.fl-kpi-label {{ color: var(--fl-muted); font-size: .78rem; letter-spacing: .02em; }}
.fl-kpi-value {{ font-size: 1.55rem; font-weight: 700; margin-top: 2px; }}
.fl-kpi-hint {{ color: var(--fl-muted); font-size: .75rem; margin-top: 3px; }}

.fl-badge {{
  display:inline-flex; align-items:center; gap:4px;
  border-radius: 999px; padding: 2px 9px; font-size: .73rem; font-weight: 600;
  border: 1px solid transparent; margin-right: 4px; white-space: nowrap;
}}
.fl-badge-fact {{ background: {COLORS['success_soft']}; color: {COLORS['success']}; }}
.fl-badge-ai {{ background: {COLORS['primary_soft']}; color: {COLORS['primary']}; }}
.fl-badge-assume {{ background: {COLORS['warning_soft']}; color: {COLORS['warning']}; }}
.fl-badge-unknown {{ background: #F2F4F7; color: {COLORS['muted']}; }}
.fl-badge-p1 {{ background: {COLORS['danger_soft']}; color: {COLORS['danger']}; }}
.fl-badge-p2 {{ background: {COLORS['warning_soft']}; color: {COLORS['warning']}; }}
.fl-badge-p3 {{ background: {COLORS['success_soft']}; color: {COLORS['success']}; }}
.fl-badge-muted {{ background: #F2F4F7; color: {COLORS['muted']}; }}

.fl-quote {{
  border-left: 3px solid {COLORS['primary']};
  background: #FAFCFE;
  padding: 8px 12px; margin: 6px 0; border-radius: 0 8px 8px 0;
  font-size: .88rem; color: #344054;
}}
.fl-quote-meta {{ color: var(--fl-muted); font-size: .74rem; margin-bottom: 2px; }}
.fl-note {{
  background: {COLORS['warning_soft']}; border: 1px solid #F7D08A;
  color: #7A4A03; border-radius: 10px; padding: 10px 13px; font-size: .84rem;
}}
.fl-empty {{
  border: 1px dashed var(--fl-border); border-radius: 12px;
  padding: 26px; text-align:center; color: var(--fl-muted); background:#fff;
}}
.fl-section {{ margin: 1.1rem 0 .2rem 0; }}
.fl-section-title {{ font-size: 1.08rem; font-weight: 650; }}
.fl-section-sub {{ color: var(--fl-muted); font-size: .8rem; }}
.fl-divider {{ height:1px; background: var(--fl-border); margin: 14px 0; }}
[data-testid="stMetricValue"] {{ font-size: 1.3rem; }}
.stButton > button, .stDownloadButton > button {{ border-radius: 8px; font-weight: 600; }}
.stButton > button[kind="primary"] {{ background: {COLORS['primary']}; border-color: {COLORS['primary']}; }}
[data-testid="stExpander"] {{ background: #fff; border: 1px solid var(--fl-border); border-radius: 10px; }}
[data-testid="stDataFrame"] {{ border: 1px solid var(--fl-border); border-radius: 10px; }}
</style>
"""


def inject() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def style_fig(fig: go.Figure, height: int = 320, showlegend: bool = False) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        font=dict(family='Inter, "Segoe UI", "Microsoft YaHei", sans-serif', size=12,
                  color=COLORS["text"]),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=showlegend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor=COLORS["border"],
                        font=dict(color=COLORS["text"])),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#EEF1F5", zeroline=False,
                     linecolor=COLORS["border"])
    fig.update_yaxes(showgrid=False, zeroline=False, linecolor=COLORS["border"])
    return fig


def product_color(name: str, idx: int = 0) -> str:
    return PRODUCT_COLORS.get(name, FALLBACK_PALETTE[idx % len(FALLBACK_PALETTE)])
