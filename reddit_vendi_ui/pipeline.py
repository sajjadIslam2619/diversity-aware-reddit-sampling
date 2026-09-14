"""Scrape a full subreddit, sample two diverse sets, and score them with Vendi."""

from __future__ import annotations

import time
from typing import Callable

import pandas as pd

from approaches import (
    EMBED_MODEL,
    encode_texts,
    label_corpus,
    sample_approach_1,
    sample_approach_2,
)
from chroma_store import embeddings_for_posts
from paths import DATA_DIR, ensure_dirs
from scrape import load_cached_posts, normalize_subreddit, scrape_cache_path, scrape_posts, save_cached_posts
from storage import save_run

Progress = Callable[[str], None]


def run_pipeline(
    subreddit: str,
    n_select: int,
    listing: str = "hot",
    time_filter: str = "all",
    progress: Progress | None = None,
    use_cache: bool = True,
) -> dict:
    progress = progress or (lambda _msg: None)
    if n_select not in (20, 50, 100):
        raise ValueError("Select 20, 50, or 100.")
    name = normalize_subreddit(subreddit)
    timings: dict[str, float] = {}
    cache_file = scrape_cache_path(name, listing, time_filter)
    from_cache = bool(use_cache and load_cached_posts(name, listing, time_filter) is not None)

    started = time.perf_counter()
    corpus = scrape_posts(
        name,
        listing=listing,
        time_filter=time_filter,
        progress=progress,
        use_cache=use_cache,
    )
    timings["scrape_s"] = time.perf_counter() - started

    started = time.perf_counter()
    labeled = label_corpus(corpus, progress=progress)
    save_cached_posts(labeled, name, listing, time_filter, overwrite=True)
    timings["label_s"] = time.perf_counter() - started

    started = time.perf_counter()
    embeddings = embeddings_for_posts(
        labeled,
        encode=encode_texts,
        subreddit=name,
        listing=listing,
        embed_model=EMBED_MODEL,
        progress=progress,
    )
    timings["embed_s"] = time.perf_counter() - started

    started = time.perf_counter()
    approach1 = sample_approach_1(labeled, embeddings, k=int(n_select), progress=progress)
    timings["approach1_s"] = time.perf_counter() - started

    started = time.perf_counter()
    approach2 = sample_approach_2(
        labeled,
        n=int(n_select),
        progress=progress,
        embeddings=embeddings,
    )
    timings["approach2_s"] = time.perf_counter() - started

    ensure_dirs()
    a1_path = cache_file.with_name(cache_file.stem.replace("_corpus", "") + f"_sample_a1_k{approach1['k_used']}.csv")
    a2_path = cache_file.with_name(cache_file.stem.replace("_corpus", "") + f"_sample_a2_n{approach2['n_selected']}.csv")
    approach1["posts"].to_csv(a1_path, index=False)
    approach2["posts"].to_csv(a2_path, index=False)
    progress(f"Saved samples to data/{a1_path.name} and data/{a2_path.name}")

    n_corpus = len(labeled)
    summary = {
        "subreddit": name,
        "listing": listing,
        "time_filter": time_filter if listing == "top" else "",
        "n_requested": int(n_select),
        "n_scraped": n_corpus,
        "n_selected_a1": len(approach1["posts"]),
        "n_selected_a2": int(approach2["n_selected"]),
        "n_label_groups": int(approach2["n_label_groups"]),
        "k_requested": int(approach1["k_requested"]),
        "k_used": int(approach1["k_used"]),
        "vendi_approach1": float(approach1["vendi"]),
        "vendi_approach2": float(approach2["vendi"]),
        "mean_sim_approach1": float(approach1["mean_similarity"]),
        "mean_sim_approach2": float(approach2["mean_similarity"]),
        "embedding_dim": int(approach1["embedding_dim"]),
        "embed_model": approach1["model"],
        "mental_model": approach2["mental_model"],
        "emotion_model": approach2["emotion_model"],
        "timings": timings,
        "posts_from_cache": from_cache,
        "data_file": str(cache_file),
        "sample_a1_file": str(a1_path),
        "sample_a2_file": str(a2_path),
        "chroma_dir": str(DATA_DIR / "chroma"),
    }
    progress("Saving this run locally…")
    combined = pd.concat(
        [
            approach1["posts"].assign(selection="approach1"),
            approach2["posts"].assign(selection="approach2"),
        ],
        ignore_index=True,
    )
    summary["run_id"] = save_run(summary, combined)
    return {
        "summary": summary,
        "corpus": labeled,
        "posts": approach1["posts"],
        "posts_a1": approach1["posts"],
        "posts_a2": approach2["posts"],
        "cluster_counts": approach1["cluster_counts"],
        "centroid_similarity": approach1["centroid_similarity"],
    }


def posts_for_export(posts: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "post_id",
        "title",
        "post_author",
        "selftext",
        "score",
        "ups",
        "downs",
        "url",
        "permalink",
        "created_utc",
        "post_word_count",
        "post_size_category",
        "mental_health_class",
        "post_emotion",
        "size_mental_emotion_label",
        "cluster",
        "comment-body-1",
        "comment-body-score-1",
        "comment-author-1",
        "comment-body-2",
        "comment-body-score-2",
        "comment-author-2",
        "comment-body-3",
        "comment-body-score-3",
        "comment-author-3",
        "data_source",
        "listing",
        "selection",
    ]
    keep = [col for col in columns if col in posts.columns]
    return posts[keep].copy()
