"""Scrape a full subreddit, sample two diverse sets, and score them with Vendi."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Callable

import pandas as pd

from approaches import (
    EMBED_MODEL,
    compute_status,
    encode_texts,
    label_corpus,
    reset_used_devices,
    sample_approach_1,
    sample_approach_2,
    used_devices,
)
from chroma_store import embeddings_for_posts
from paths import DATA_DIR, ensure_dirs
from scrape import (
    load_raw_csv,
    normalize_subreddit,
    scrape_and_save,
)
from storage import save_run

Progress = Callable[[str], None]


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


def _load_corpus(
    subreddit: str,
    listing: str,
    time_filter: str,
    progress: Progress,
    use_cache: bool,
    corpus: pd.DataFrame | None,
    raw_path: str | None,
    scrape_kwargs: dict,
) -> tuple[pd.DataFrame, Path | None, bool, str]:
    name = normalize_subreddit(subreddit)
    if corpus is not None:
        progress(f"Using {len(corpus)} already scraped posts.")
        return corpus.copy(), Path(raw_path) if raw_path else None, True, name
    if raw_path:
        loaded = load_raw_csv(Path(raw_path))
        if loaded is None:
            raise RuntimeError(f"Could not load raw file {raw_path}.")
        progress(f"Using {len(loaded)} posts from data/{Path(raw_path).name}")
        return loaded, Path(raw_path), True, name
    scraped = scrape_and_save(
        name,
        listing=listing,
        time_filter=time_filter,
        progress=progress,
        use_cache=use_cache,
        **scrape_kwargs,
    )
    return scraped["posts"], scraped["path"], bool(scraped["from_cache"]), name


def _embed_corpus(
    frame: pd.DataFrame,
    subreddit: str,
    listing: str,
    progress: Progress,
):
    progress(f"Checking ChromaDB for r/{subreddit} embeddings…")
    return embeddings_for_posts(
        frame,
        encode=encode_texts,
        subreddit=subreddit,
        listing=listing,
        embed_model=EMBED_MODEL,
        progress=progress,
    )


def _sample_stem(saved_raw: Path | None, name: str, n: int) -> str:
    if saved_raw:
        return saved_raw.stem.replace("_raw", "")
    return f"{name}_{n}"


def run_scrape(
    subreddit: str,
    listing: str = "hot",
    time_filter: str = "all",
    progress: Progress | None = None,
    use_cache: bool = True,
    raw_file: str | Path | None = None,
    total_posts: int = 1000,
    batch_size: int = 100,
    rate_limit_sleep_s: int = 60,
    include_stickied: bool = False,
    require_selftext: bool = True,
    comment_sort: str = "top",
    comments_limit: int = 3,
) -> dict:
    progress = progress or (lambda _msg: None)
    result = scrape_and_save(
        subreddit,
        listing=listing,
        time_filter=time_filter,
        progress=progress,
        use_cache=use_cache,
        raw_file=raw_file,
        total_posts=total_posts,
        batch_size=batch_size,
        rate_limit_sleep_s=rate_limit_sleep_s,
        include_stickied=include_stickied,
        require_selftext=require_selftext,
        comment_sort=comment_sort,
        comments_limit=comments_limit,
    )
    path = result["path"]
    result["relpath"] = f"data/{path.name}"
    return result


def run_approach_1(
    subreddit: str,
    n_select: int,
    listing: str = "hot",
    time_filter: str = "all",
    progress: Progress | None = None,
    use_cache: bool = True,
    corpus: pd.DataFrame | None = None,
    raw_path: str | None = None,
    **scrape_kwargs,
) -> dict:
    progress = progress or (lambda _msg: None)
    reset_used_devices()
    if n_select not in (10, 20, 50, 100):
        raise ValueError("Select 10, 20, 50, or 100.")
    timings: dict[str, float] = {}
    started = time.perf_counter()
    frame, saved_raw, from_cache, name = _load_corpus(
        subreddit, listing, time_filter, progress, use_cache, corpus, raw_path, scrape_kwargs
    )
    timings["scrape_s"] = time.perf_counter() - started

    started = time.perf_counter()
    embeddings = _embed_corpus(frame, name, listing, progress)
    timings["embed_s"] = time.perf_counter() - started

    started = time.perf_counter()
    approach1 = sample_approach_1(frame, embeddings, k=int(n_select), progress=progress)
    timings["approach1_s"] = time.perf_counter() - started

    ensure_dirs()
    stem = _sample_stem(saved_raw, name, len(frame))
    a1_path = DATA_DIR / f"{name}_{approach1['k_used']}_a1.csv"
    approach1["posts"].to_csv(a1_path, index=False)
    progress(f"Saved cluster centroids sample to data/{a1_path.name}")

    summary = {
        "subreddit": name,
        "listing": listing,
        "time_filter": time_filter if listing == "top" else "",
        "n_requested": int(n_select),
        "n_scraped": len(frame),
        "n_selected_a1": len(approach1["posts"]),
        "k_requested": int(approach1["k_requested"]),
        "k_used": int(approach1["k_used"]),
        "vendi_approach1": _as_float(approach1.get("vendi")),
        "mean_sim_approach1": _as_float(approach1.get("mean_similarity")),
        "embedding_dim": int(approach1["embedding_dim"]),
        "embed_model": approach1["model"],
        "timings": timings,
        "posts_from_cache": from_cache,
        "data_file": str(saved_raw) if saved_raw else str(DATA_DIR),
        "sample_a1_file": str(a1_path),
        "chroma_dir": str(DATA_DIR / "chroma"),
        "compute": compute_status()["caption"],
        "compute_used": used_devices(),
    }
    summary["run_id"] = save_run(
        {
            **summary,
            "vendi_approach2": None,
            "mean_sim_approach2": None,
        },
        approach1["posts"].assign(selection="approach1"),
    )
    return {
        "summary": summary,
        "vendi": summary["vendi_approach1"],
        "corpus": approach1.get("corpus", frame),
        "posts": approach1["posts"],
        "cluster_counts": approach1["cluster_counts"],
        "centroid_similarity": approach1["centroid_similarity"],
    }


def run_approach_2(
    subreddit: str,
    n_select: int,
    listing: str = "hot",
    time_filter: str = "all",
    progress: Progress | None = None,
    use_cache: bool = True,
    corpus: pd.DataFrame | None = None,
    raw_path: str | None = None,
    **scrape_kwargs,
) -> dict:
    progress = progress or (lambda _msg: None)
    reset_used_devices()
    if n_select not in (10, 20, 50, 100):
        raise ValueError("Select 10, 20, 50, or 100.")
    timings: dict[str, float] = {}
    started = time.perf_counter()
    frame, saved_raw, from_cache, name = _load_corpus(
        subreddit, listing, time_filter, progress, use_cache, corpus, raw_path, scrape_kwargs
    )
    timings["scrape_s"] = time.perf_counter() - started

    started = time.perf_counter()
    labeled = label_corpus(frame, progress=progress)
    timings["label_s"] = time.perf_counter() - started

    started = time.perf_counter()
    embeddings = _embed_corpus(labeled, name, listing, progress)
    timings["embed_s"] = time.perf_counter() - started

    started = time.perf_counter()
    approach2 = sample_approach_2(
        labeled,
        n=int(n_select),
        progress=progress,
        embeddings=embeddings,
    )
    timings["approach2_s"] = time.perf_counter() - started

    ensure_dirs()
    stem = _sample_stem(saved_raw, name, len(labeled))
    a2_path = DATA_DIR / f"{stem}_sample_a2_n{approach2['n_selected']}.csv"
    approach2["posts"].to_csv(a2_path, index=False)
    progress(f"Saved label coverage sample to data/{a2_path.name}")

    summary = {
        "subreddit": name,
        "listing": listing,
        "time_filter": time_filter if listing == "top" else "",
        "n_requested": int(n_select),
        "n_scraped": len(labeled),
        "n_selected_a2": int(approach2["n_selected"]),
        "n_label_groups": int(approach2["n_label_groups"]),
        "k_requested": int(n_select),
        "k_used": int(approach2["n_selected"]),
        "vendi_approach2": _as_float(approach2.get("vendi")),
        "mean_sim_approach2": _as_float(approach2.get("mean_similarity")),
        "embed_model": EMBED_MODEL,
        "mental_model": approach2["mental_model"],
        "emotion_model": approach2["emotion_model"],
        "timings": timings,
        "posts_from_cache": from_cache,
        "data_file": str(saved_raw) if saved_raw else str(DATA_DIR),
        "sample_a2_file": str(a2_path),
        "chroma_dir": str(DATA_DIR / "chroma"),
        "compute": compute_status()["caption"],
        "compute_used": used_devices(),
    }
    summary["run_id"] = save_run(
        {
            **summary,
            "vendi_approach1": None,
            "mean_sim_approach1": None,
        },
        approach2["posts"].assign(selection="approach2"),
    )
    return {
        "summary": summary,
        "vendi": summary["vendi_approach2"],
        "corpus": labeled,
        "posts": approach2["posts"],
    }


def run_pipeline(
    subreddit: str,
    n_select: int,
    listing: str = "hot",
    time_filter: str = "all",
    progress: Progress | None = None,
    use_cache: bool = True,
    corpus: pd.DataFrame | None = None,
    raw_path: str | None = None,
    **scrape_kwargs,
) -> dict:
    a1 = run_approach_1(
        subreddit,
        n_select,
        listing=listing,
        time_filter=time_filter,
        progress=progress,
        use_cache=use_cache,
        corpus=corpus,
        raw_path=raw_path,
        **scrape_kwargs,
    )
    a2 = run_approach_2(
        subreddit,
        n_select,
        listing=listing,
        time_filter=time_filter,
        progress=progress,
        use_cache=use_cache,
        corpus=a1.get("corpus", corpus),
        raw_path=raw_path,
        **scrape_kwargs,
    )
    summary = {
        **a1["summary"],
        **a2["summary"],
        "vendi_approach1": a1["summary"]["vendi_approach1"],
        "vendi_approach2": a2["summary"]["vendi_approach2"],
        "mean_sim_approach1": a1["summary"]["mean_sim_approach1"],
        "mean_sim_approach2": a2["summary"]["mean_sim_approach2"],
        "n_selected_a1": a1["summary"]["n_selected_a1"],
        "n_selected_a2": a2["summary"]["n_selected_a2"],
        "k_used": a1["summary"]["k_used"],
        "sample_a1_file": a1["summary"]["sample_a1_file"],
        "sample_a2_file": a2["summary"]["sample_a2_file"],
    }
    return {
        "summary": summary,
        "corpus": a2.get("corpus", a1.get("corpus")),
        "posts": a1["posts"],
        "posts_a1": a1["posts"],
        "posts_a2": a2["posts"],
        "cluster_counts": a1["cluster_counts"],
        "centroid_similarity": a1["centroid_similarity"],
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
        "selection_role",
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
