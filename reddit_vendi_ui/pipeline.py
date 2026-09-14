"""Scrape a subreddit and score Approach 1 and Approach 2."""

from __future__ import annotations

import time
from typing import Callable

import pandas as pd

from approaches import run_approach_1, run_approach_2
from scrape import normalize_subreddit, scrape_posts
from storage import save_run

Progress = Callable[[str], None]


def run_pipeline(
    subreddit: str,
    n_posts: int,
    k: int,
    listing: str = "hot",
    time_filter: str = "all",
    progress: Progress | None = None,
) -> dict:
    progress = progress or (lambda _msg: None)
    name = normalize_subreddit(subreddit)
    timings: dict[str, float] = {}

    started = time.perf_counter()
    posts = scrape_posts(
        name,
        limit=int(n_posts),
        listing=listing,
        time_filter=time_filter,
        progress=progress,
    )
    timings["scrape_s"] = time.perf_counter() - started

    started = time.perf_counter()
    approach2 = run_approach_2(posts, progress=progress)
    timings["approach2_s"] = time.perf_counter() - started

    started = time.perf_counter()
    approach1 = run_approach_1(approach2["posts"], k=int(k), progress=progress)
    timings["approach1_s"] = time.perf_counter() - started

    merged = approach1["posts"]
    n = len(merged)
    summary = {
        "subreddit": name,
        "listing": listing,
        "time_filter": time_filter if listing == "top" else "",
        "n_requested": int(n_posts),
        "n_scraped": n,
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
    }
    progress("Saving this run locally…")
    summary["run_id"] = save_run(summary, merged)
    return {
        "summary": summary,
        "posts": merged,
        "cluster_counts": approach1["cluster_counts"],
        "centroid_similarity": approach1["centroid_similarity"],
    }


def posts_for_export(posts: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "post_id",
        "title",
        "selftext",
        "score",
        "url",
        "created_utc",
        "post_word_count",
        "post_size_category",
        "mental_health_class",
        "post_emotion",
        "size_mental_emotion_label",
        "cluster",
        "comment_body",
        "comment_score",
        "data_source",
        "listing",
    ]
    keep = [col for col in columns if col in posts.columns]
    return posts[keep].copy()
