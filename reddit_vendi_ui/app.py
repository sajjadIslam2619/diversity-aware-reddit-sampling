"""Reddit scrape UI: Approach 1 vs Approach 2 Vendi scores.

Run from the repo root or from this folder:

    python -m streamlit run reddit_vendi_ui/app.py
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from approaches import EMBED_MODEL, EMOTION_MODEL, MENTAL_MODEL
from paths import DATA_DIR, MODELS_DIR
from pipeline import posts_for_export, run_pipeline
from scrape import SIZE_BANDS, scrape_cache_path
from storage import list_runs

WORKFLOW_PNG = ROOT.parent / "reddit_data_proces" / "reddit_etl_workflow.png"

st.set_page_config(
    page_title="Reddit Vendi compare",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.4rem; max-width: 1180px; }
      .hero h1 {
        font-size: 2.05rem;
        letter-spacing: -0.03em;
        margin-bottom: 0.15rem;
      }
      .hero p { color: #57534e; font-size: 1.02rem; margin-top: 0; }
      .score-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.9rem;
        margin: 0.4rem 0 1rem 0;
      }
      .card {
        border: 1px solid #e7e5e4;
        border-radius: 16px;
        padding: 1.05rem 1.15rem 0.95rem;
        background: #fffdf8;
      }
      .card.a1 { border-top: 4px solid #0f766e; }
      .card.a2 { border-top: 4px solid #c2410c; }
      .kicker {
        font-size: 0.75rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #78716c;
        font-weight: 650;
      }
      .card h3 { margin: 0.2rem 0 0.55rem; font-size: 1.05rem; }
      .score {
        font-size: 2.7rem;
        font-weight: 680;
        letter-spacing: -0.04em;
        line-height: 1;
        font-variant-numeric: tabular-nums;
      }
      .card.a1 .score { color: #0f766e; }
      .card.a2 .score { color: #c2410c; }
      .sub { color: #57534e; margin-top: 0.45rem; font-size: 0.92rem; }
      @media (max-width: 800px) {
        .score-grid { grid-template-columns: 1fr; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def _missing_deps() -> list[str]:
    missing = []
    for module, package in (
        ("praw", "praw"),
        ("dotenv", "python-dotenv"),
        ("sklearn", "scikit-learn"),
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("sentence_transformers", "sentence-transformers"),
        ("chromadb", "chromadb"),
    ):
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    return missing


def _score_cards(v1: float, v2: float, n: int, k_used: int) -> str:
    def card(kind: str, kicker: str, title: str, score: float, detail: str) -> str:
        ratio = 0.0 if n <= 0 else 100.0 * score / n
        return (
            f'<div class="card {kind}">'
            f'<div class="kicker">{html.escape(kicker)}</div>'
            f"<h3>{html.escape(title)}</h3>"
            f'<div class="score">{score:.2f}</div>'
            f'<div class="sub">{html.escape(detail)} · {ratio:.0f}% of maximum ({n})</div>'
            "</div>"
        )

    left = card(
        "a1",
        "Approach 1",
        "One post nearest each of k clusters",
        v1,
        f"Cosine Vendi · k={k_used} clusters",
    )
    right = card(
        "a2",
        "Approach 2",
        "Size × mental status × emotion coverage",
        v2,
        "Cosine Vendi on the selected set",
    )
    return f'<div class="score-grid">{left}{right}</div>'


def _posts_table(posts: pd.DataFrame) -> None:
    show_cols = [
        col
        for col in (
            "created_utc",
            "title",
            "post_author",
            "score",
            "post_word_count",
            "post_size_category",
            "mental_health_class",
            "post_emotion",
            "cluster",
            "permalink",
        )
        if col in posts.columns
    ]
    if "permalink" not in posts.columns and "url" in posts.columns:
        show_cols.append("url")
    show = posts[show_cols].rename(
        columns={
            "created_utc": "created",
            "post_author": "author",
            "post_word_count": "words",
            "post_size_category": "size",
            "mental_health_class": "mental status",
            "post_emotion": "emotion",
            "permalink": "post",
        }
    )
    link_col = "post" if "post" in show.columns else "url"
    st.dataframe(
        show,
        use_container_width=True,
        hide_index=True,
        column_config={link_col: st.column_config.LinkColumn("post")} if link_col in show.columns else None,
    )


def _render_results(result: dict) -> None:
    summary = result["summary"]
    posts_a1 = result.get("posts_a1", result["posts"])
    posts_a2 = result.get("posts_a2", result["posts"])
    corpus = result.get("corpus")
    n_corpus = int(summary["n_scraped"])
    n1 = int(summary.get("n_selected_a1") or len(posts_a1))
    n2 = int(summary.get("n_selected_a2") or len(posts_a2))
    n_score = max(n1, n2)
    v1 = float(summary["vendi_approach1"])
    v2 = float(summary["vendi_approach2"])

    st.markdown(
        _score_cards(v1, v2, n_score, int(summary["k_used"])),
        unsafe_allow_html=True,
    )

    gap = v1 - v2
    if abs(gap) < 0.05:
        compare = "The two selected sets are about equally diverse under cosine Vendi."
    elif gap > 0:
        compare = "Approach 1’s cluster sample is more diverse (higher cosine Vendi)."
    else:
        compare = "Approach 2’s label-coverage sample is more diverse (higher cosine Vendi)."
    cache_note = (
        " Reused corpus from data/."
        if summary.get("posts_from_cache")
        else " Fresh scrape saved to data/."
    )
    st.caption(
        f"r/{summary['subreddit']} · {summary['listing']} · "
        f"corpus {n_corpus} posts · selected {n1} vs {n2} · run #{summary['run_id']}.{cache_note} {compare} "
        "Both scores use the same cosine Vendi on selftext embeddings of the selected posts."
    )

    chart = pd.DataFrame(
        {"Vendi score": [v1, v2]},
        index=["Approach 1", "Approach 2"],
    )
    st.bar_chart(chart, height=220)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Corpus posts", str(n_corpus))
    c2.metric("Mean cosine (A1 sample)", f"{summary['mean_sim_approach1']:.3f}")
    c3.metric("Mean cosine (A2 sample)", f"{summary['mean_sim_approach2']:.3f}")
    c4.metric("Clusters k", str(summary["k_used"]))

    tabs = st.tabs(["Approach 1 sample", "Approach 2 sample", "Corpus", "Run details"])
    with tabs[0]:
        st.write("One post nearest each cluster centroid")
        _posts_table(posts_a1)
        st.dataframe(result["cluster_counts"], use_container_width=True, hide_index=True)
        st.write("Cosine similarity of cluster centers")
        st.dataframe(result["centroid_similarity"].round(3), use_container_width=True)
        st.download_button(
            "Download Approach 1 sample",
            data=posts_for_export(posts_a1).to_csv(index=False).encode("utf-8"),
            file_name=f"r_{summary['subreddit']}_a1_k{summary['k_used']}.csv",
            mime="text/csv",
            key="dl-a1",
        )

    with tabs[1]:
        st.write("Posts spread across size × mental status × emotion")
        _posts_table(posts_a2)
        left, right = st.columns(2)
        with left:
            st.write("Post size")
            st.dataframe(
                posts_a2["post_size_category"].value_counts().rename_axis("size").reset_index(name="posts"),
                hide_index=True,
                use_container_width=True,
            )
            st.write("Mental status")
            st.dataframe(
                posts_a2["mental_health_class"].value_counts().rename_axis("mental status").reset_index(name="posts"),
                hide_index=True,
                use_container_width=True,
            )
        with right:
            st.write("Emotion")
            st.dataframe(
                posts_a2["post_emotion"].value_counts().rename_axis("emotion").reset_index(name="posts"),
                hide_index=True,
                use_container_width=True,
            )
        st.write("Size × mental status")
        st.dataframe(
            pd.crosstab(posts_a2["post_size_category"], posts_a2["mental_health_class"]),
            use_container_width=True,
        )
        st.download_button(
            "Download Approach 2 sample",
            data=posts_for_export(posts_a2).to_csv(index=False).encode("utf-8"),
            file_name=f"r_{summary['subreddit']}_a2_n{n2}.csv",
            mime="text/csv",
            key="dl-a2",
        )

    with tabs[2]:
        if corpus is None or corpus.empty:
            st.info("Corpus table was not returned for this run.")
        else:
            st.write(f"Full scraped listing ({len(corpus)} posts)")
            _posts_table(corpus)
            st.download_button(
                "Download corpus CSV",
                data=posts_for_export(corpus).to_csv(index=False).encode("utf-8"),
                file_name=f"r_{summary['subreddit']}_corpus.csv",
                mime="text/csv",
                key="dl-corpus",
            )

    with tabs[3]:
        timings = summary.get("timings") or {}
        st.write(
            {
                "subreddit": summary["subreddit"],
                "listing": summary["listing"],
                "time_filter": summary.get("time_filter") or "—",
                "corpus_posts": n_corpus,
                "selected": {"approach1": n1, "approach2": n2},
                "label_groups": summary.get("n_label_groups"),
                "models": {
                    "embed": summary["embed_model"],
                    "mental": summary["mental_model"],
                    "emotion": summary["emotion_model"],
                    "folder": str(MODELS_DIR),
                },
                "data_file": summary.get("data_file") or str(DATA_DIR),
                "sample_a1_file": summary.get("sample_a1_file"),
                "sample_a2_file": summary.get("sample_a2_file"),
                "chroma": summary.get("chroma_dir"),
                "posts_from_cache": bool(summary.get("posts_from_cache")),
                "seconds": {key: round(value, 1) for key, value in timings.items()},
                "sqlite": "reddit_vendi_ui/output/runs.sqlite",
            }
        )


with st.sidebar:
    st.header("How scoring works")
    st.markdown(
        """
**Approach 1** embeds the full corpus (ChromaDB), clusters into k=20/50/100, and keeps one post nearest each centroid.

**Approach 2** labels every post (mental status, emotion, size) and picks 20/50/100 posts that cover those label combinations.

Vendi then scores the two selected sets in the same embedding space. Higher means the sample is more diverse.
        """
    )
    bands = ", ".join(f"{name} {low}–{high}" for name, low, high in SIZE_BANDS)
    st.caption(f"Size bands: {bands} words (selftext). Outside that range is out_of_band.")
    history = list_runs()
    if not history.empty:
        st.subheader("Recent runs")
        st.dataframe(
            history.rename(
                columns={
                    "vendi_approach1": "Vendi 1",
                    "vendi_approach2": "Vendi 2",
                    "n_scraped": "corpus",
                    "k_used": "k",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )

st.markdown(
    """
    <div class="hero">
      <h1>Reddit diversity compare</h1>
      <p>Scrape a subreddit, sample diverse posts two ways, then compare those samples with Vendi.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

missing = _missing_deps()
if missing:
    st.error(
        "Install missing packages, then restart the app: "
        + "pip install "
        + " ".join(missing)
    )
    st.stop()

controls, notes = st.columns([1.15, 0.85], gap="large")
with controls:
    subreddit = st.text_input(
        "Subreddit",
        value="OpiatesRecovery",
        help="Name, r/name, or a reddit.com/r/name URL.",
    )
    n_posts = st.radio(
        "Sample size / Approach 1 clusters (k)",
        [20, 50, 100],
        horizontal=True,
        index=0,
        help="Approach 1 uses this as the number of clusters (one post per cluster). Approach 2 selects this many posts across labels.",
    )
    listing = st.selectbox(
        "Listing",
        ["hot", "top", "new"],
        format_func=lambda value: {
            "hot": "Hot",
            "top": "Top",
            "new": "New (most recent)",
        }[value],
    )
    time_filter = "all"
    if listing == "top":
        time_filter = st.selectbox(
            "Top window",
            ["day", "week", "month", "year", "all"],
            index=2,
        )
    force_scrape = st.checkbox(
        "Re-scrape even if data/ already has this listing",
        value=False,
        help="Leave unchecked to reuse the saved corpus in data/ and embeddings in data/chroma.",
    )
    run_clicked = st.button("Scrape corpus and sample", type="primary")

with notes:
    cache_hint = ""
    try:
        cache_file = scrape_cache_path(subreddit, listing, time_filter)
        if cache_file.is_file():
            cache_hint = f" Cached posts: `data/{cache_file.name}`."
    except Exception:
        cache_hint = ""
    st.markdown(
        f"""
**Fixed from the existing notebooks**

- Embeddings: `{EMBED_MODEL}` on `selftext`, stored in ChromaDB under `data/chroma`
- Mental status: `{MENTAL_MODEL}` on `selftext`
- Emotion: `{EMOTION_MODEL}` on `selftext`
- Top 3 comments saved as `comment-body-1` / `comment-body-score-1`, then 2 and 3

The app scrapes the full listing (Reddit usually caps near 1000), then selects k posts two ways. Corpus, samples, and models are reused from `data/` and `models/` when present.{cache_hint}
Credentials are read from the repo `.env`. This package does not change the notebooks.
        """
    )

if run_clicked:
    try:
        with st.status("Running both approaches…", expanded=True) as status:
            def _progress(message: str) -> None:
                status.write(message)

            result = run_pipeline(
                subreddit=subreddit,
                n_select=int(n_posts),
                listing=listing,
                time_filter=time_filter,
                progress=_progress,
                use_cache=not force_scrape,
            )
            status.update(label="Scores ready", state="complete", expanded=False)
        st.session_state["result"] = result
    except Exception as exc:
        st.error(str(exc))
        st.info(
            "Adjust the subreddit, listing, or credentials in the repo .env, then run again. "
            "Nothing was written into the existing notebooks or CSVs."
        )

result = st.session_state.get("result")
if result:
    selected = max(
        int(result["summary"].get("n_selected_a1") or 0),
        int(result["summary"].get("n_selected_a2") or 0),
    )
    if selected < int(result["summary"]["n_requested"]):
        st.warning(
            f"The corpus has {result['summary']['n_scraped']} usable posts, "
            f"fewer than the {result['summary']['n_requested']} requested, "
            "so both samples use every post Reddit returned."
        )
    _render_results(result)
elif WORKFLOW_PNG.is_file():
    st.image(str(WORKFLOW_PNG), caption="Both approaches run on the same scraped posts.")
else:
    st.info("Choose a subreddit, then scrape the corpus and sample.")
