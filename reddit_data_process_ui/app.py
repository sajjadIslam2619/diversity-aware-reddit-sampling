"""Cluster centroids vs label coverage Vendi scores.

Run from the repo root or from this folder:

    python -m streamlit run reddit_data_process_ui/app.py
"""

from __future__ import annotations

import importlib
import html
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import math

import pandas as pd
import streamlit as st

# Streamlit keeps old module objects on rerun; reload so new helpers exist.
import vendi as vendi_mod
import approaches as approaches_mod
import storage as storage_mod
import scrape as scrape_mod
import chroma_store as chroma_mod
import pipeline as pipeline_mod

importlib.reload(vendi_mod)
importlib.reload(approaches_mod)
importlib.reload(storage_mod)
importlib.reload(scrape_mod)
importlib.reload(chroma_mod)
importlib.reload(pipeline_mod)

from approaches import EMBED_MODEL, EMOTION_MODEL, MENTAL_MODEL, compute_status
from paths import DATA_DIR, MODELS_DIR
from scrape import SIZE_BANDS, list_raw_files, scrape_and_save as run_scrape
from chroma_store import embedding_coverage
from storage import list_runs
from pipeline import posts_for_export, run_approach_1, run_approach_2

WORKFLOW_PNG = ROOT.parent / "reddit_data_proces" / "reddit_etl_workflow.png"
A1_NAME = "Cluster centroids"
A2_NAME = "Label coverage"

st.set_page_config(
    page_title="Reddit Post Selection Pipeline",
    page_icon="🔍",
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
      .device-pill {
        display: inline-block;
        margin-left: 0.4rem;
        padding: 0.12rem 0.6rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 650;
        letter-spacing: 0.02em;
        vertical-align: middle;
      }
      .device-pill.gpu { background: #ccfbf1; color: #0f766e; }
      .device-pill.cpu { background: #ffedd5; color: #c2410c; }
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


def _score_cards(v1: float | None, v2: float | None, n: int, k_used: int) -> str:
    def card(kind: str, kicker: str, title: str, score: float | None, detail: str) -> str:
        score_txt = "—" if score is None else f"{score:.2f}"
        if score is None or n <= 0:
            ratio_txt = html.escape(detail)
        else:
            ratio_txt = f"{html.escape(detail)} · {100.0 * score / n:.0f}% of maximum ({n})"
        return (
            f'<div class="card {kind}">'
            f'<div class="kicker">{html.escape(kicker)}</div>'
            f"<h3>{html.escape(title)}</h3>"
            f'<div class="score">{score_txt}</div>'
            f'<div class="sub">{ratio_txt}</div>'
            "</div>"
        )

    left = card(
        "a1",
        A1_NAME,
        "One post nearest each of k clusters",
        v1,
        f"Cosine Vendi · k={k_used} clusters",
    )
    right = card(
        "a2",
        A2_NAME,
        "Size × mental status × emotion coverage",
        v2,
        "Cosine Vendi on the selected set",
    )
    return f'<div class="score-grid">{left}{right}</div>'


def _compute_info() -> dict:
    return compute_status()


def _compute_pill_html(info: dict | None = None) -> str:
    info = info or _compute_info()
    kind = "gpu" if info["cuda"] else "cpu"
    return (
        f'<span class="device-pill {kind}">'
        f'Using {html.escape(info["caption"])}'
        "</span>"
    )


def _render_scrape(scrape: dict) -> None:
    n = int(scrape["n"])
    filename = scrape.get("filename") or Path(scrape["path"]).name
    relpath = scrape.get("relpath") or f"data/{filename}"
    if scrape.get("from_cache"):
        st.success(f"Loaded **{n}** posts from `{filename}`.")
    else:
        st.success(f"Collected **{n}** posts and saved `{filename}`.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Posts collected", str(n))
    c2.metric("File name", filename)
    c3.metric("Folder", "data/")
    st.caption(f"Full path: `{relpath}` · r/{scrape.get('subreddit', '')} · {scrape.get('listing', '')}")
    posts = scrape.get("posts")
    if posts is not None and not posts.empty:
        st.write("Raw scrape preview")
        _posts_table(posts)
        st.download_button(
            "Download raw CSV",
            data=posts.to_csv(index=False).encode("utf-8"),
            file_name=filename,
            mime="text/csv",
            key="dl-raw",
        )


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
            "selection_role",
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


def _as_float(value, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    try:
        if math.isnan(number) or math.isinf(number):
            return default
    except TypeError:
        return default
    return number


def _vendi_metric(label: str, score: float | None, n: int, extra: str) -> None:
    if score is None:
        st.metric(label, "n/a")
        st.caption(extra)
        return
    st.metric(label, f"{score:.2f}", help=extra)
    if n:
        st.caption(f"{extra} · {100.0 * score / n:.0f}% of maximum ({n})")
    else:
        st.caption(extra)


def _render_approach_1(result: dict) -> None:
    summary = result["summary"]
    posts = result["posts"]
    n = int(summary.get("n_selected_a1") or len(posts) or 0)
    vendi = _as_float(summary.get("vendi_approach1"))
    if vendi is None:
        vendi = _as_float(result.get("vendi"))
    st.markdown(f"### {A1_NAME}")
    if vendi is None:
        st.success(
            f"Created **{n}** clusters and kept **{n}** centroid posts "
            "(the post nearest each cluster center)."
        )
        st.warning("Vendi score could not be computed for this sample.")
    else:
        st.success(
            f"Created **{n}** clusters and kept **{n}** centroid posts "
            f"(the post nearest each cluster center). Vendi: **{vendi:.2f}**"
        )
    _vendi_metric(f"{A1_NAME} Vendi", vendi, n, f"Cosine Vendi · {n} centroid posts")
    mean_sim = _as_float(summary.get("mean_sim_approach1"))
    mean_txt = "—" if mean_sim is None else f"{mean_sim:.3f}"
    save_name = f"{summary['subreddit']}_{n}_a1.csv"
    saved_path = summary.get("sample_a1_file")
    st.caption(
        f"r/{summary['subreddit']} · {n} centroid posts · mean cosine {mean_txt} · "
        f"embeddings from ChromaDB (`data/chroma`)"
    )
    _posts_table(posts)
    if result.get("cluster_counts") is not None:
        st.write("Posts per cluster in the corpus (one centroid kept from each)")
        st.dataframe(result["cluster_counts"], use_container_width=True, hide_index=True)
    if result.get("centroid_similarity") is not None:
        st.write("Cosine similarity of cluster centers")
        st.dataframe(result["centroid_similarity"].round(3), use_container_width=True)

    save_col, download_col = st.columns(2)
    with save_col:
        if st.button(f"Save {n} posts to data/{save_name}", key="save-a1-disk"):
            path = DATA_DIR / save_name
            posts_for_export(posts).to_csv(path, index=False)
            st.session_state["a1_saved_file"] = save_name
            st.success(f"Saved {n} centroid posts to data/{save_name}")
    with download_col:
        st.download_button(
            f"Download {A1_NAME} CSV",
            data=posts_for_export(posts).to_csv(index=False).encode("utf-8"),
            file_name=save_name,
            mime="text/csv",
            key="dl-a1",
        )
    shown = st.session_state.get("a1_saved_file") or (Path(saved_path).name if saved_path else None)
    if shown:
        st.caption(f"File: `data/{shown}`")


def _render_approach_2(result: dict) -> None:
    summary = result["summary"]
    posts = result["posts"]
    n = int(summary.get("n_selected_a2") or len(posts))
    vendi = _as_float(summary.get("vendi_approach2"))
    if vendi is None:
        vendi = _as_float(result.get("vendi"))
    st.markdown(f"### {A2_NAME}")
    if vendi is None:
        st.success(f"Selected **{n}** posts across label groups.")
        st.warning("Vendi score could not be computed for this sample.")
    else:
        st.success(f"Vendi score: **{vendi:.2f}**  ·  {n} posts across label groups")
    _vendi_metric(f"{A2_NAME} Vendi", vendi, n, "Cosine Vendi on size × mental × emotion coverage")
    mean_sim = _as_float(summary.get("mean_sim_approach2"))
    mean_txt = "—" if mean_sim is None else f"{mean_sim:.3f}"
    st.caption(
        f"r/{summary['subreddit']} · {n} selected · {summary.get('n_label_groups', '—')} label groups · "
        f"mean cosine {mean_txt}"
    )
    _posts_table(posts)
    if "post_size_category" in posts.columns:
        left, right = st.columns(2)
        with left:
            st.write("Post size")
            st.dataframe(
                posts["post_size_category"].value_counts().rename_axis("size").reset_index(name="posts"),
                hide_index=True,
                use_container_width=True,
            )
            if "mental_health_class" in posts.columns:
                st.write("Mental status")
                st.dataframe(
                    posts["mental_health_class"].value_counts().rename_axis("mental status").reset_index(name="posts"),
                    hide_index=True,
                    use_container_width=True,
                )
        with right:
            if "post_emotion" in posts.columns:
                st.write("Emotion")
                st.dataframe(
                    posts["post_emotion"].value_counts().rename_axis("emotion").reset_index(name="posts"),
                    hide_index=True,
                    use_container_width=True,
                )
    st.download_button(
        f"Download {A2_NAME} CSV",
        data=posts_for_export(posts).to_csv(index=False).encode("utf-8"),
        file_name=f"{summary['subreddit']}_{n}_a2.csv",
        mime="text/csv",
        key="dl-a2",
    )


def _render_compare(a1: dict, a2: dict) -> None:
    s1 = a1["summary"]
    s2 = a2["summary"]
    n1 = int(s1.get("n_selected_a1") or len(a1["posts"]))
    n2 = int(s2.get("n_selected_a2") or len(a2["posts"]))
    v1 = _as_float(s1.get("vendi_approach1"))
    if v1 is None:
        v1 = _as_float(a1.get("vendi"))
    v2 = _as_float(s2.get("vendi_approach2"))
    if v2 is None:
        v2 = _as_float(a2.get("vendi"))
    if v1 is None or v2 is None:
        return
    st.markdown("### Compare")
    st.markdown(_score_cards(v1, v2, max(n1, n2), int(s1["k_used"])), unsafe_allow_html=True)
    #chart = pd.DataFrame({"Vendi score": [v1, v2]}, index=[A1_NAME, A2_NAME])
    #st.bar_chart(chart, height=220)


def _render_results(result: dict) -> None:
    summary = result["summary"]
    posts_a1 = result.get("posts_a1", result["posts"])
    posts_a2 = result.get("posts_a2", result["posts"])
    corpus = result.get("corpus")
    n_corpus = int(summary["n_scraped"])
    n1 = int(summary.get("n_selected_a1") or len(posts_a1))
    n2 = int(summary.get("n_selected_a2") or len(posts_a2))
    n_score = max(n1, n2)
    v1 = _as_float(summary.get("vendi_approach1"))
    v2 = _as_float(summary.get("vendi_approach2"))
    if v1 is None or v2 is None:
        if v1 is not None:
            _render_approach_1(result)
        if v2 is not None:
            _render_approach_2(result)
        return

    st.markdown(
        _score_cards(v1, v2, n_score, int(summary["k_used"])),
        unsafe_allow_html=True,
    )

    gap = v1 - v2
    if abs(gap) < 0.05:
        compare = "The two selected sets are about equally diverse under cosine Vendi."
    elif gap > 0:
        compare = f"{A1_NAME} is more diverse (higher cosine Vendi)."
    else:
        compare = f"{A2_NAME} is more diverse (higher cosine Vendi)."
    used = summary.get("compute_used") or {}
    if used:
        used_bits = ", ".join(f"{step} on {device}" for step, device in used.items())
        compute_note = f" Models ran on: {used_bits}."
    else:
        compute_note = f" Models available on {summary.get('compute') or _compute_info()['caption']} (this run reused cached labels/embeddings)."
    cache_note = (
        " Reused corpus from data/."
        if summary.get("posts_from_cache")
        else " Fresh scrape saved to data/."
    )
    st.caption(
        f"r/{summary['subreddit']} · {summary['listing']} · "
        f"corpus {n_corpus} posts · selected {n1} vs {n2} · run #{summary['run_id']}."
        f"{cache_note}{compute_note} {compare} "
        "Both scores use the same cosine Vendi on selftext embeddings of the selected posts."
    )

    chart = pd.DataFrame(
        {"Vendi score": [v1, v2]},
        index=[A1_NAME, A2_NAME],
    )
    st.bar_chart(chart, height=220)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Corpus posts", str(n_corpus))
    mean1 = _as_float(summary.get("mean_sim_approach1"))
    mean2 = _as_float(summary.get("mean_sim_approach2"))
    c2.metric(f"Mean cosine ({A1_NAME})", "n/a" if mean1 is None else f"{mean1:.3f}")
    c3.metric(f"Mean cosine ({A2_NAME})", "n/a" if mean2 is None else f"{mean2:.3f}")
    c4.metric("Clusters k", str(summary["k_used"]))

    tabs = st.tabs([A1_NAME, A2_NAME, "Corpus", "Run details"])
    with tabs[0]:
        st.write("One post nearest each cluster centroid")
        _posts_table(posts_a1)
        st.dataframe(result["cluster_counts"], use_container_width=True, hide_index=True)
        st.write("Cosine similarity of cluster centers")
        st.dataframe(result["centroid_similarity"].round(3), use_container_width=True)
        st.download_button(
            f"Download {A1_NAME} CSV",
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
            f"Download {A2_NAME} CSV",
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
                "selected": {A1_NAME: n1, A2_NAME: n2},
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
                "compute": summary.get("compute") or _compute_info()["caption"],
                "models_ran_on": summary.get("compute_used") or {},
                "posts_from_cache": bool(summary.get("posts_from_cache")),
                "seconds": {key: round(value, 1) for key, value in timings.items()},
                "sqlite": "reddit_data_process_ui/output/runs.sqlite",
            }
        )


with st.sidebar:
    st.header("How scoring works")
    #st.markdown(_compute_pill_html(), unsafe_allow_html=True)
    st.markdown(
        """
**Cluster centroids** embeds the full corpus, makes n clusters (10/20/50/100), and keeps the centroid post of each cluster so you get n posts.

**Label coverage** labels every post (mental status, emotion, size) and picks 10/20/50/100 posts that cover those label combinations.

Vendi then scores the two selected sets in the same embedding space. Higher means the sample is more diverse.
        """
    )
    bands = ", ".join(f"{name} {low}–{high}" for name, low, high in SIZE_BANDS)
    st.caption(f"Size bands: {bands} words (selftext). Outside that range is out_of_band.")
    history = list_runs()
    if not history.empty:
        st.subheader("Recent runs")
        show = history.rename(
            columns={
                "vendi_approach1": "Centroids",
                "vendi_approach2": "Coverage",
                "n_scraped": "corpus",
                "k_used": "k",
            }
        )
        for col in ("Centroids", "Coverage"):
            if col in show.columns:
                show[col] = show[col].map(
                    lambda value: (lambda number: f"{number:.2f}" if number is not None else "—")(
                        _as_float(value)
                    )
                )
        st.dataframe(
            show,
            hide_index=True,
            use_container_width=True,
        )

missing = _missing_deps()
if missing:
    st.error(
        "Install missing packages, then restart the app: "
        + "pip install "
        + " ".join(missing)
    )
    st.stop()

_compute = _compute_info()
st.markdown(
    """
    <div class="hero">
      <h1>Reddit Post Diversity Compare</h1>
      <p>First scrape a subreddit and save a raw CSV, then sample with cluster centroids or label coverage and compare them with Vendi.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.subheader("1. Scrape posts")
controls, notes = st.columns([1.15, 0.85], gap="large")
with controls:
    subreddit = st.text_input(
        "Subreddit",
        value="OpiatesRecovery",
        help="Name, r/name, or a reddit.com/r/name URL.",
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
    total_posts = st.select_slider(
        "Max listing posts to fetch",
        options=[100, 200, 300, 500, 1000],
        value=1000,
        help="Reddit usually returns at most about 1000 listing items.",
    )
    batch_col, pause_col = st.columns(2)
    with batch_col:
        batch_size = st.selectbox("Batch size", [25, 50, 100], index=2)
    with pause_col:
        rate_limit_sleep_s = st.selectbox(
            "Pause between batches (seconds)",
            [0, 30, 60],
            index=2,
        )
    include_stickied = st.checkbox("Include stickied posts", value=False)
    require_selftext = st.checkbox("Only keep posts with selftext", value=True)
    comment_sort = st.selectbox("Comment sort", ["top", "best", "new"], index=0)
    comments_limit = st.slider("Comments to save per post", 0, 3, 3)
    saved_files = []
    try:
        if subreddit.strip():
            saved_files = list_raw_files(subreddit)
    except Exception:
        saved_files = []
    selected_raw = None
    if saved_files:
        selected_raw = st.selectbox(
            "Saved raw files for this subreddit",
            saved_files,
            format_func=lambda path: path.name,
        )
    force_scrape = st.checkbox(
        "Re-scrape even if a raw file already exists",
        value=False,
        help="Leave unchecked to load the selected saved file.",
    )
    scrape_clicked = st.button("Scrape and save", type="primary")

with notes:
    st.markdown(
        f"""
**Raw file naming**

Files are saved under `data/` as `subreddit_postcount_raw.csv`.
Example: `OpiatesRecovery_847_raw.csv`.

**Scrape options**

- Fetch in batches, then pause so Reddit rate limits are respected
- Skip stickied / empty posts unless you turn those filters off
- Comments are optional; 0 skips comment requests

Embeddings (`{EMBED_MODEL}`), mental status (`{MENTAL_MODEL}`), and emotion (`{EMOTION_MODEL}`) run later, when you sample.
Credentials come from the repo `.env`.
        """
    )
    if saved_files:
        st.caption("Already in `data/`: " + ", ".join(path.name for path in saved_files[:6]))

if scrape_clicked:
    try:
        with st.status("Scraping posts…", expanded=True) as status:
            def _scrape_progress(message: str) -> None:
                status.write(message)

            scrape = run_scrape(
                subreddit=subreddit,
                listing=listing,
                time_filter=time_filter,
                progress=_scrape_progress,
                use_cache=not force_scrape,
                raw_file=None if force_scrape else selected_raw,
                total_posts=int(total_posts),
                batch_size=int(batch_size),
                rate_limit_sleep_s=int(rate_limit_sleep_s),
                include_stickied=include_stickied,
                require_selftext=require_selftext,
                comment_sort=comment_sort,
                comments_limit=int(comments_limit),
            )
            status.update(
                label=f"Saved {scrape['n']} posts as {scrape['filename']}",
                state="complete",
                expanded=False,
            )
        st.session_state["scrape"] = scrape
        st.session_state.pop("approach1", None)
        st.session_state.pop("approach2", None)
    except Exception as exc:
        st.error(str(exc))
        st.info(
            "Adjust the subreddit, listing, or credentials in the repo .env, then scrape again."
        )

scrape = st.session_state.get("scrape")
if scrape:
    _render_scrape(scrape)

    st.subheader(f"2. Run {A1_NAME} or {A2_NAME}")
    n_posts = st.radio(
        f"Sample size / {A1_NAME} clusters (k)",
        [10, 20, 50, 100],
        horizontal=True,
        index=0,
        help=f"{A1_NAME} builds this many clusters and keeps the centroid post of each (n posts). {A2_NAME} selects this many posts across labels.",
    )
    try:
        coverage = embedding_coverage(scrape["posts"], scrape["subreddit"])
        if coverage["stored"]:
            st.info(
                f"ChromaDB already has **{coverage['stored']} / {coverage['corpus']}** "
                f"embeddings for r/{coverage['subreddit']}. "
                + (
                    f"{coverage['missing']} new posts will be encoded and saved."
                    if coverage["missing"]
                    else "No new embedding work is needed."
                )
            )
        else:
            st.caption(
                f"No embeddings stored yet for r/{scrape['subreddit']}. "
                "The first approach you run will encode posts and save them in data/chroma."
            )
    except Exception as exc:
        st.caption(f"ChromaDB status unavailable ({exc}). Embeddings will be created when you run an approach.")

    if _compute["cuda"]:
        st.caption(f"Models will use GPU: {_compute['name']}")
    else:
        st.caption("Models will use CPU. Install a CUDA build of PyTorch to run on GPU.")

    run_a1, run_a2 = st.columns(2)
    with run_a1:
        a1_clicked = st.button(f"Run {A1_NAME}", type="primary", use_container_width=True)
    with run_a2:
        a2_clicked = st.button(f"Run {A2_NAME}", type="primary", use_container_width=True)

    shared_kwargs = dict(
        subreddit=scrape["subreddit"],
        n_select=int(n_posts),
        listing=scrape.get("listing") or listing,
        time_filter=scrape.get("time_filter") or time_filter,
        corpus=scrape["posts"],
        raw_path=str(scrape["path"]),
    )

    if a1_clicked:
        try:
            with st.status(f"{A1_NAME} on {_compute['using']}…", expanded=True) as status:
                def _progress(message: str) -> None:
                    status.write(message)
                    if "retrying on CPU" in message:
                        status.update(label="Running on CPU (GPU out of memory)…", state="running")

                result_a1 = run_approach_1(progress=_progress, **shared_kwargs)
                vendi = _as_float(result_a1["summary"].get("vendi_approach1"))
                if vendi is None:
                    vendi = _as_float(result_a1.get("vendi"))
                status.update(
                    label=(
                        f"{A1_NAME} Vendi {vendi:.2f}"
                        if vendi is not None
                        else f"{A1_NAME} complete"
                    ),
                    state="complete",
                    expanded=False,
                )
            st.session_state["approach1"] = result_a1
        except Exception as exc:
            st.error(str(exc))
            st.info(f"{A1_NAME} uses the saved raw posts and ChromaDB embeddings for this subreddit.")

    if a2_clicked:
        try:
            with st.status(f"{A2_NAME} on {_compute['using']}…", expanded=True) as status:
                def _progress_a2(message: str) -> None:
                    status.write(message)
                    if "retrying on CPU" in message:
                        status.update(label="Running on CPU (GPU out of memory)…", state="running")

                result_a2 = run_approach_2(progress=_progress_a2, **shared_kwargs)
                vendi = _as_float(result_a2["summary"].get("vendi_approach2"))
                if vendi is None:
                    vendi = _as_float(result_a2.get("vendi"))
                status.update(
                    label=(
                        f"{A2_NAME} Vendi {vendi:.2f}"
                        if vendi is not None
                        else f"{A2_NAME} complete"
                    ),
                    state="complete",
                    expanded=False,
                )
            st.session_state["approach2"] = result_a2
        except Exception as exc:
            st.error(str(exc))
            st.info(f"{A2_NAME} labels the saved raw posts, then reuses ChromaDB embeddings for this subreddit.")

    approach1 = st.session_state.get("approach1")
    approach2 = st.session_state.get("approach2")
    if approach1:
        _render_approach_1(approach1)
    if approach2:
        _render_approach_2(approach2)
    if approach1 and approach2:
        _render_compare(approach1, approach2)
elif WORKFLOW_PNG.is_file():
    st.image(str(WORKFLOW_PNG), caption=f"Scrape first, then run {A1_NAME} or {A2_NAME} on the saved posts.")
else:
    st.info("Choose a subreddit and scrape options, then save a raw CSV.")
