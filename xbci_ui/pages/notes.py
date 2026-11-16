"""Notes & Blog: articles, videos, glossary, references."""
from pathlib import Path

import streamlit as st

from .. import content
from ..theme import callout, chip, hero

VIDEO_DIR = Path(__file__).resolve().parents[2] / "assets" / "videos"
VIDEO_TITLES = {"01_bci_pipeline": "From brain signal to decision (the BCI pipeline)",
                "02_erd_motor_imagery": "Motor imagery and ERD: why the rhythm drops",
                "03_csp_spatial_filtering": "How spatial filtering (CSP) pulls out a pattern",
                "04_explainability": "How we explain a model: occlusion and attribution"}


def render(S):
    hero("Notes & Blog", "Plain-language articles that build up from <b>what EEG is</b> to <b>how each model computes its answer</b> and <b>how we explain it</b>. "
         "Read in order, or jump to what you need.", [chip("8 articles", "blue"), chip("glossary", "gray"), chip("references", "gray"), chip("videos", "gray")])
    t1, t2, t3, t4 = st.tabs(["📝 Articles", "🎬 Videos & animations", "📖 Glossary", "🔗 References"])
    with t1:
        keys = list(content.ARTICLES)
        c1, c2 = st.columns([1, 3])
        with c1:
            sel = st.radio("Articles", keys, format_func=lambda k: content.ARTICLES[k]["title"], key="art_sel", label_visibility="collapsed")
        with c2:
            a = content.ARTICLES[sel]
            st.markdown(f"## {a['title']}")
            st.markdown(f"{chip(str(a['minutes']) + ' min read', 'gray')} &nbsp; <span class='small-muted'>{a['summary']}</span>", unsafe_allow_html=True)
            st.markdown(a["body"])
            i = keys.index(sel)
            st.caption(f"Article {i + 1} of {len(keys)}" + (f" · next: {content.ARTICLES[keys[i + 1]]['title']}" if i + 1 < len(keys) else ""))
    with t2:
        vids = sorted(VIDEO_DIR.glob("*.mp4")) if VIDEO_DIR.exists() else []
        if vids:
            for v in vids:
                st.markdown(f"#### {VIDEO_TITLES.get(v.stem, v.stem.replace('_', ' ').title())}")
                st.video(str(v))
        else:
            callout("tip", "No video files found", f"Drop `.mp4` files into `{VIDEO_DIR}` and they appear here automatically.")
        st.markdown("#### Interactive animations in this app")
        st.markdown("- **Start here → Interactive tour**: press *Play* to watch a real trial travel from raw signal to decision.\n"
                    "- **Playground → Time–frequency editor**: change a rhythm in a time window and watch the prediction react.\n"
                    "- **Science Q&A → noise robustness / mirror test**: live experiments on the test set.")
    with t3:
        q = st.text_input("Search the glossary", key="gl_q")
        for term, desc in content.GLOSSARY.items():
            if not q or q.lower() in term.lower() or q.lower() in desc.lower():
                st.markdown(f"**{term}** — {desc}")
    with t4:
        st.markdown("Key sources behind the methods and numbers in this app (please verify details against the originals before citing):")
        for r in content.REFERENCES:
            st.markdown(f"- {r}")
