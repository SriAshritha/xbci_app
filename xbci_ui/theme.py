"""Look & feel: CSS, header, cards, step-flow, callouts."""
import html
import streamlit as st

PRIMARY = "#1a56ff"
INK = "#1b2433"
MUTED = "#5b6678"
BG = "#f0f3f8"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stApp {{ font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif; color: {INK}; }}
.stApp {{ background: {BG}; }}
header[data-testid="stHeader"] {{ background: transparent; height: 0; }}
#MainMenu, footer {{ visibility: hidden; }}
.block-container {{ padding-top: 0.6rem; padding-bottom: 4rem; max-width: 1320px; }}
section[data-testid="stSidebar"] {{ background: #ffffff; border-right: 1px solid #e3e8f0; }}
h1, h2, h3, h4 {{ letter-spacing: -0.01em; color: {INK}; }}

/* top bar */
.xb-brand {{ display:flex; align-items:center; gap:.7rem; padding:.35rem 0; }}
.xb-logo {{ width:38px; height:38px; border-radius:10px; background:linear-gradient(135deg,{PRIMARY},#6f3bff);
  display:flex; align-items:center; justify-content:center; color:#fff; font-weight:800; font-size:1.05rem; }}
.xb-brand b {{ font-size:1.02rem; line-height:1.15; display:block; }}
.xb-brand span {{ font-size:.78rem; color:{MUTED}; }}
.xb-chip {{ display:inline-block; padding:.22rem .65rem; border-radius:999px; font-size:.74rem; font-weight:600;
  background:#fff8e6; border:1px solid #f3d58a; color:#8a5a00; }}
.xb-chip.ok {{ background:#eaf8f0; border-color:#a6dcbc; color:#14683a; }}
.xb-chip.blue {{ background:#e9f0ff; border-color:#bcd0ff; color:#1a45c7; }}
.xb-chip.gray {{ background:#eef1f6; border-color:#dde3ec; color:#4a566a; }}

/* nav pills (st.radio with key=nav) */
.st-key-nav div[role="radiogroup"] {{ gap:.25rem; background:#fff; border:1px solid #e3e8f0; padding:.25rem; border-radius:12px;
  justify-content:center; flex-wrap:wrap; }}
.st-key-nav label {{ background:transparent; padding:.38rem .9rem; border-radius:9px; cursor:pointer; margin:0; }}
.st-key-nav label > div:first-child {{ display:none; }}
.st-key-nav label p {{ font-size:.84rem; font-weight:600; color:{MUTED}; margin:0; }}
.st-key-nav label:has(input:checked) {{ background:#e9f0ff; }}
.st-key-nav label:has(input:checked) p {{ color:{PRIMARY}; }}
.st-key-nav label:hover {{ background:#f3f6fb; }}

/* cards */
div[data-testid="stVerticalBlockBorderWrapper"] {{ background:#fff; border-radius:14px; }}
.xb-card {{ background:#fff; border:1px solid #e3e8f0; border-radius:14px; padding:1.1rem 1.3rem; margin-bottom:.8rem; }}
.xb-card h4 {{ margin:0 0 .35rem 0; }}
.xb-hero {{ background:linear-gradient(135deg,#ffffff 0%, #eaf0ff 100%); border:1px solid #dbe5ff; border-radius:18px;
  padding:1.8rem 2rem; margin:.6rem 0 1rem 0; }}
.xb-hero h1 {{ font-size:2rem; margin:0 0 .4rem 0; font-weight:800; }}
.xb-hero p {{ color:{MUTED}; font-size:1.02rem; margin:0; max-width:900px; }}
.xb-kpi {{ background:#fff; border:1px solid #e3e8f0; border-radius:14px; padding:.9rem 1.1rem; }}
.xb-kpi .v {{ font-size:1.6rem; font-weight:800; color:{INK}; line-height:1.15; }}
.xb-kpi .l {{ font-size:.76rem; color:{MUTED}; text-transform:uppercase; letter-spacing:.05em; font-weight:600; }}
.xb-kpi .s {{ font-size:.78rem; color:{MUTED}; }}
.xb-kpi .v.long {{ font-size:1.02rem; word-break:break-all; overflow-wrap:anywhere; line-height:1.3; }}
.xb-kpi {{ overflow:hidden; }}
.xb-chip {{ max-width:100%; word-break:break-all; }}

/* step flow */
.xb-step {{ display:flex; gap:.85rem; align-items:flex-start; margin:.2rem 0 .5rem 0; }}
.xb-step .n {{ min-width:34px; height:34px; border-radius:50%; background:{PRIMARY}; color:#fff; font-weight:700;
  display:flex; align-items:center; justify-content:center; font-size:.95rem; box-shadow:0 2px 6px rgba(26,86,255,.3); }}
.xb-step .t {{ font-size:1.12rem; font-weight:700; line-height:1.2; }}
.xb-step .d {{ color:{MUTED}; font-size:.9rem; margin-top:.15rem; }}

/* callouts */
.xb-call {{ border-radius:12px; padding:.8rem 1rem; margin:.5rem 0; font-size:.92rem; border:1px solid; }}
.xb-call.tip {{ background:#eef4ff; border-color:#cddcff; }}
.xb-call.warn {{ background:#fff7e8; border-color:#f4d9a0; }}
.xb-call.ok {{ background:#ecf9f1; border-color:#b4e2c6; }}
.xb-call.bad {{ background:#fdeeee; border-color:#f3c0c0; }}
.xb-call b.h {{ display:block; margin-bottom:.15rem; }}

/* verdict */
.xb-verdict {{ border-radius:14px; padding:1rem 1.3rem; font-size:1.05rem; font-weight:600; border:1px solid; margin:.3rem 0 .8rem 0; }}
.xb-verdict.agree {{ background:#ecf9f1; border-color:#b4e2c6; color:#14683a; }}
.xb-verdict.disagree {{ background:#fff4e5; border-color:#f4d9a0; color:#8a5a00; }}

/* arch flow */
.xb-arch {{ display:flex; flex-wrap:wrap; gap:.4rem; align-items:center; }}
.xb-block {{ background:#fff; border:1px solid #cfd8e6; border-left:4px solid {PRIMARY}; border-radius:8px; padding:.35rem .6rem; font-size:.76rem; }}
.xb-block small {{ color:{MUTED}; display:block; }}
.xb-arrow {{ color:#9aa6ba; font-size:1.1rem; }}

.stButton>button[kind="primary"] {{ background:{PRIMARY}; border:none; border-radius:9px; font-weight:600; }}
.stButton>button {{ border-radius:9px; font-weight:600; }}
div[data-testid="stFileUploaderDropzone"] {{ border:2px dashed #c6d2e6; border-radius:14px; background:#fff; }}
.stTabs [data-baseweb="tab"] {{ font-weight:600; }}
.small-muted {{ color:{MUTED}; font-size:.82rem; }}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def esc(t):
    return html.escape(str(t))


def chip(text, kind=""):
    return f'<span class="xb-chip {kind}">{esc(text)}</span>'


def header(right_html=""):
    c1, c2, c3 = st.columns([2.3, 5.6, 2.1], vertical_alignment="center")
    with c1:
        st.markdown('<div class="xb-brand"><div class="xb-logo">X</div><div><b><span style="color:#1a45c7">XBCI</span> '
                    'Explainable BCI</b><span>Motor-imagery EEG · EEGNet vs CSP+LDA</span></div></div>',
                    unsafe_allow_html=True)
    return c2, c3


def hero(title, text, chips=()):
    ch = " ".join(chips)
    st.markdown(f'<div class="xb-hero"><h1>{title}</h1><p>{text}</p><div style="margin-top:.8rem">{ch}</div></div>',
                unsafe_allow_html=True)


def kpi(label, value, sub=""):
    import re
    long_cls = " long" if len(re.sub(r"<[^>]+>", "", str(value))) > 11 else ""
    return (f'<div class="xb-kpi"><div class="l">{esc(label)}</div><div class="v{long_cls}">{value}</div>'
            f'<div class="s">{sub}</div></div>')


def kpi_row(items):
    cols = st.columns(len(items))
    for c, (l, v, s) in zip(cols, items):
        c.markdown(kpi(l, v, s), unsafe_allow_html=True)


def callout(kind, title, body):
    st.markdown(f'<div class="xb-call {kind}"><b class="h">{title}</b>{body}</div>', unsafe_allow_html=True)


def step_header(n, title, desc=""):
    st.markdown(f'<div class="xb-step"><div class="n">{n}</div><div><div class="t">{title}</div>'
                f'<div class="d">{desc}</div></div></div>', unsafe_allow_html=True)


def arch_flow(blocks):
    """blocks: list of (title, subtitle)."""
    parts = []
    for i, (t, s) in enumerate(blocks):
        parts.append(f'<div class="xb-block"><b>{esc(t)}</b><small>{esc(s)}</small></div>')
        if i < len(blocks) - 1:
            parts.append('<div class="xb-arrow">➜</div>')
    st.markdown('<div class="xb-arch">' + "".join(parts) + "</div>", unsafe_allow_html=True)


# ----------------------------------------------------------------------------- version-safe widgets
def plot(fig, key=None):
    cfg = {"displaylogo": False, "toImageButtonOptions": {"scale": 2}}
    try:
        st.plotly_chart(fig, width="stretch", key=key, config=cfg)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, key=key, config=cfg)


def table(df, hide_index=True):
    try:
        st.dataframe(df, width="stretch", hide_index=hide_index)
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=hide_index)
