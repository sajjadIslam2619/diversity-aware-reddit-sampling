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
from pipeline import posts_for_export, run_pipeline
from scrape import SIZE_BANDS
from storage import list_runs

WORKFLOW_PNG = ROOT.parent / "reddit_data_proces" / "reddit_etl_workflow.png"
SUGGESTED_K = {20: 4, 50: 7, 100: 14}

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
        "Embed, cluster, similarity",
        v1,
        f"Cosine kernel · k={k_used}",
    )
    right = card(
        "a2",
        "Approach 2",
        "Mental status, emotion, post size",
        v2,
        "Category-match kernel",
    )
    return f'<div class="score-grid">{left}{right}</div>'


def _render_results(result: dict) -> None:
    summary = result["summary"]
    posts = result["posts"]
    n = int(summary["n_scraped"])
    v1 = float(summary["vendi_approach1"])
    v2 = float(summary["vendi_approach2"])

    st.markdown(
        _score_cards(v1, v2, n, int(summary["k_used"])),
        unsafe_allow_html=True,
    )

    gap = v1 - v2
    if abs(gap) < 0.05:
        compare = "The two kernels see about the same diversity in this selection."
    elif gap > 0:
        compare = (
            "Approach 1’s embedding kernel sees more diversity than Approach 2’s "
            "size / mental-status / emotion labels."
        )
    else:
        compare = (
            "Approach 2’s label kernel sees more diversity than Approach 1’s "
            "embedding kernel."
        )
    st.caption(
        f"r/{summary['subreddit']} · {summary['listing']} · "
        f"{n} of {summary['n_requested']} posts · run #{summary['run_id']}. {compare} "
        "Both scores are effective sample sizes from 1 (all alike) to N (mutually dissimilar)."
    )

    chart = pd.DataFrame(
        {"Vendi score": [v1, v2]},
        index=["Approach 1", "Approach 2"],
    )
    st.bar_chart(chart, height=220)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean cosine similarity", f"{summary['mean_sim_approach1']:.3f}")
    c2.metric("Mean label overlap", f"{summary['mean_sim_approach2']:.3f}")
    c3.metric("Clusters used", str(summary["k_used"]))
    c4.metric("Embedding dim", str(summary["embedding_dim"]))

    tabs = st.tabs(["Posts", "Approach 1", "Approach 2", "Run details"])
    with tabs[0]:
        show = posts[
            [
                "created_utc",
                "title",
                "score",
                "post_word_count",
                "post_size_category",
                "mental_health_class",
                "post_emotion",
                "cluster",
                "url",
            ]
        ].rename(
            columns={
                "created_utc": "created",
                "post_word_count": "words",
                "post_size_category": "size",
                "mental_health_class": "mental status",
                "post_emotion": "emotion",
            }
        )
        st.dataframe(
            show,
            use_container_width=True,
            hide_index=True,
            column_config={"url": st.column_config.LinkColumn("post")},
        )
        export = posts_for_export(posts)
        st.download_button(
            "Download posts CSV",
            data=export.to_csv(index=False).encode("utf-8"),
            file_name=f"r_{summary['subreddit']}_{n}_vendi.csv",
            mime="text/csv",
        )

    with tabs[1]:
        st.write("Posts per cluster")
        st.dataframe(result["cluster_counts"], use_container_width=True, hide_index=True)
        st.write("Cosine similarity of cluster centers")
        sim = result["centroid_similarity"].round(3)
        st.dataframe(sim, use_container_width=True)
        examples = (
            posts.groupby("cluster", sort=True)["title"]
            .first()
            .rename("example title")
            .reset_index()
        )
        st.write("Example title from each cluster")
        st.dataframe(examples, use_container_width=True, hide_index=True)

    with tabs[2]:
        left, right = st.columns(2)
        with left:
            st.write("Post size")
            st.dataframe(
                posts["post_size_category"].value_counts().rename_axis("size").reset_index(name="posts"),
                hide_index=True,
                use_container_width=True,
            )
            st.write("Mental status")
            st.dataframe(
                posts["mental_health_class"].value_counts().rename_axis("mental status").reset_index(name="posts"),
                hide_index=True,
                use_container_width=True,
            )
        with right:
            st.write("Emotion")
            st.dataframe(
                posts["post_emotion"].value_counts().rename_axis("emotion").reset_index(name="posts"),
                hide_index=True,
                use_container_width=True,
            )
        st.write("Size × mental status")
        cross = pd.crosstab(posts["post_size_category"], posts["mental_health_class"])
        st.dataframe(cross, use_container_width=True)
        out_of_band = int((posts["post_size_category"] == "out_of_band").sum())
        if out_of_band:
            st.info(
                f"{out_of_band} posts are outside the 10–500 word bands "
                "(labeled out_of_band). They stay in the Vendi score so both approaches use the same posts."
            )

    with tabs[3]:
        timings = summary.get("timings") or {}
        st.write(
            {
                "subreddit": summary["subreddit"],
                "listing": summary["listing"],
                "time_filter": summary.get("time_filter") or "—",
                "models": {
                    "embed": summary["embed_model"],
                    "mental": summary["mental_model"],
                    "emotion": summary["emotion_model"],
                },
                "seconds": {key: round(value, 1) for key, value in timings.items()},
                "sqlite": "reddit_vendi_ui/output/runs.sqlite",
            }
        )


with st.sidebar:
    st.header("How scoring works")
    st.markdown(
        """
**Approach 1** embeds each post, clusters those vectors, and scores diversity with a cosine-similarity Vendi score.

**Approach 2** labels mental status, emotion, and post size, then scores diversity with a category-match Vendi score.

Same scraped posts. Different similarity. Higher score means more dissimilar posts.
        """
    )
    bands = ", ".join(f"{name} {low}–{high}" for name, low, high in SIZE_BANDS)
    st.caption(f"Size bands: {bands} words (title + body). Outside that range is out_of_band.")
    history = list_runs()
    if not history.empty:
        st.subheader("Recent runs")
        st.dataframe(
            history.rename(
                columns={
                    "vendi_approach1": "Vendi 1",
                    "vendi_approach2": "Vendi 2",
                    "n_scraped": "posts",
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
      <p>Scrape a subreddit, then score the same posts with both pipelines.</p>
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
        "How many posts",
        [20, 50, 100],
        horizontal=True,
        index=0,
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
    k = st.number_input(
        "Approach 1 clusters (k)",
        min_value=2,
        max_value=int(n_posts),
        value=SUGGESTED_K[int(n_posts)],
        step=1,
        key=f"k-{n_posts}",
        help="Suggested k keeps clusters from getting tiny. The notebooks used k=14 on a much larger corpus.",
    )
    run_clicked = st.button("Scrape and compare", type="primary")

with notes:
    st.markdown(
        f"""
**Fixed from the existing notebooks**

- Embeddings: `{EMBED_MODEL}`
- Mental status: `{MENTAL_MODEL}`
- Emotion: `{EMOTION_MODEL}`
- Best comment stored with the post; authors are not saved

The first run downloads the models. On CPU, 50–100 posts can take several minutes.
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
                n_posts=int(n_posts),
                k=int(k),
                listing=listing,
                time_filter=time_filter,
                progress=_progress,
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
    if int(result["summary"]["n_scraped"]) < int(result["summary"]["n_requested"]):
        st.warning(
            f"Reddit returned {result['summary']['n_scraped']} usable posts, "
            f"fewer than the {result['summary']['n_requested']} requested. "
            "Scores use the posts that were returned."
        )
    _render_results(result)
elif WORKFLOW_PNG.is_file():
    st.image(str(WORKFLOW_PNG), caption="Both approaches run on the same scraped posts.")
else:
    st.info("Choose a subreddit and how many posts, then scrape.")
