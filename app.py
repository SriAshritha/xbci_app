"""XBCI - Explainable Brain-Computer Interface (Streamlit app).  Run:  streamlit run app.py"""
import sys
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="XBCI · Explainable BCI", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from xbci_ui import store, theme  # noqa: E402

theme.inject_css()

PAGES = ["🏠 Start here", "🧪 Playground", "⚖️ Compare models", "🔬 Under the hood", "📊 Performance", "❓ Science Q&A", "📚 Notes & Blog", "ℹ️ About"]


def sidebar():
    sb = st.sidebar
    sb.markdown('<div class="xb-brand"><div class="xb-logo">X</div><div><b>XBCI</b><span>Explainable BCI</span></div></div>', unsafe_allow_html=True)
    sb.markdown("##### Artifacts")
    default = str(store.auto_find_root() or "")
    path = sb.text_input("Artifact folder", value=default, key="root_path", help="The XBCI_BNCI2014_001 folder produced by the Colab notebook (unzipped). You can also set the environment variable XBCI_ARTIFACT_ROOT.")
    root = store.resolve_root(path)
    if root is None:
        return None, None, None
    xc = store.get_core(str(root))
    subs = store.list_subjects(str(root))
    if not subs:
        sb.error("No subject with both models complete.")
        return root, None, None
    sid = sb.selectbox("Subject", subs, format_func=lambda s: f"Subject {s:02d}", key="subject",
                       help="Both models were trained separately for every subject.")
    sb.markdown("##### How much detail?")
    level = sb.radio("Explanation level", ["Beginner", "Intermediate", "Expert"], index=1, key="level", label_visibility="collapsed",
                     help="Beginner: plain language & analogies. Intermediate: terminology. Expert: equations & implementation details.")
    sb.caption({"Beginner": "Plain language, analogies, no maths.", "Intermediate": "Standard BCI terminology.", "Expert": "Equations and implementation details."}[level])
    sb.divider()
    try:
        m = store.get_metrics(str(root), sid)
        sb.markdown(f"**Subject {sid:02d} · test accuracy**  \n{theme.chip('EEGNet ' + format(m['eegnet_test']['accuracy'], '.0%'), 'blue')} {theme.chip('CSP+LDA ' + format(m['csp_lda_test']['accuracy'], '.0%'), '')}  \n<span class='small-muted'>chance = 25 %</span>", unsafe_allow_html=True)
    except Exception:
        pass
    sb.markdown(f"<span class='small-muted'>device: {store.get_device_str()}</span>", unsafe_allow_html=True)
    return root, sid, xc


def setup_screen():
    theme.hero("Welcome to XBCI", "Point the app to the artifact folder produced by the Colab notebook to get started.", [theme.chip("no artifacts found yet", "")])
    st.markdown("""
**1.** Download the zip from Colab/Drive and **unzip** it.  
**2.** Paste the path of the unzipped folder (the one that contains `models/`, `reference/`, `experiments/` …) into the **Artifact folder** box in the sidebar — or set the environment variable `XBCI_ARTIFACT_ROOT` before launching.  
**3.** Want to try the interface first? Generate a small *synthetic* demo folder:
```
python tools/make_demo_artifacts.py --out demo_artifacts
```
then paste `demo_artifacts/XBCI_BNCI2014_001`.
""")


def main():
    root, sid, xc = sidebar()
    if root is None or sid is None:
        c2, c3 = theme.header()
        setup_screen()
        return
    c2, c3 = theme.header()
    with c2:
        page = st.radio("Navigate", PAGES, horizontal=True, key="nav", label_visibility="collapsed")
    with c3:
        st.markdown(f'<div style="text-align:right">{theme.chip("● artifacts loaded", "ok")} {theme.chip(f"Subject {sid:02d}", "blue")}</div>', unsafe_allow_html=True)
    st.markdown("<hr style='margin:.3rem 0 .6rem 0;border:none;border-top:1px solid #e3e8f0'>", unsafe_allow_html=True)
    rs = str(root)
    S = dict(root=rs, sid=sid, ref=store.get_reference(rs), cn=list(xc.CLASS_NAMES), level=st.session_state.get("level", "Intermediate"), xc=xc)
    if page.endswith("Start here"):
        from xbci_ui.pages import home as pg
    elif page.endswith("Playground"):
        from xbci_ui.pages import playground as pg
    elif page.endswith("Compare models"):
        from xbci_ui.pages import compare as pg
    elif page.endswith("Under the hood"):
        from xbci_ui.pages import inspector as pg
    elif page.endswith("Performance"):
        from xbci_ui.pages import performance as pg
    elif page.endswith("Science Q&A"):
        from xbci_ui.pages import science as pg
    elif page.endswith("Notes & Blog"):
        from xbci_ui.pages import notes as pg
    else:
        from xbci_ui.pages import about as pg
    pg.render(S)


main()
