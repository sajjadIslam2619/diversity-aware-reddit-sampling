"""Scrape a subreddit with PRAW. Reads credentials from the repo .env."""

from __future__ import annotations

import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from paths import DATA_DIR, REPO_ROOT, ensure_dirs

SIZE_BANDS = (
    ("small", 10, 50),
    ("medium", 51, 200),
    ("large", 201, 500),
)
TOP_COMMENTS = 3
BATCH_SIZE = 100
TOTAL_POSTS = 1000
RATE_LIMIT_SLEEP_S = 60
CACHE_REQUIRED_COLS = {
    "title",
    "post_author",
    "selftext",
    "score",
    "url",
}


def load_repo_env() -> None:
    load_dotenv(REPO_ROOT / ".env")
    load_dotenv()


def normalize_subreddit(name: str) -> str:
    raw = (name or "").strip()
    lowered = raw.lower()
    for prefix in (
        "https://www.reddit.com/r/",
        "https://reddit.com/r/",
        "http://www.reddit.com/r/",
        "http://reddit.com/r/",
        "www.reddit.com/r/",
        "reddit.com/r/",
        "r/",
    ):
        if lowered.startswith(prefix):
            raw = raw[len(prefix) :]
            break
    raw = raw.strip("/").split("/")[0].strip()
    if not raw or not re.fullmatch(r"[A-Za-z0-9_]{2,21}", raw):
        raise ValueError(
            "Enter a subreddit name like OpiatesRecovery or r/OpiatesRecovery."
        )
    return raw


def word_count(text: str) -> int:
    return len(str(text).split())


def diversity_text(selftext: str) -> str:
    text = str(selftext or "").strip()
    return text if text else "[no text]"


def size_category(n_words: int) -> str:
    for label, low, high in SIZE_BANDS:
        if low <= n_words <= high:
            return label
    return "out_of_band"


def _reddit_client():
    load_repo_env()
    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT")
    missing = [
        name
        for name, value in (
            ("REDDIT_CLIENT_ID", client_id),
            ("REDDIT_CLIENT_SECRET", client_secret),
            ("REDDIT_USER_AGENT", user_agent),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(
            "Missing Reddit credentials in the repo .env: " + ", ".join(missing)
        )

    import praw

    kwargs = {
        "client_id": client_id,
        "client_secret": client_secret,
        "user_agent": user_agent,
        "check_for_updates": False,
    }
    username = os.getenv("REDDIT_USERNAME")
    password = os.getenv("REDDIT_PASSWORD")
    if username and password:
        kwargs["username"] = username
        kwargs["password"] = password
    return praw.Reddit(**kwargs)


def _empty_comment() -> dict:
    return {"body": "", "score": "", "author": ""}


def top_comments(submission, comment_sort: str = "top", limit: int = TOP_COMMENTS) -> list[dict]:
    try:
        submission.comment_sort = comment_sort
        submission.comments.replace_more(limit=0)
        raw = list(submission.comments[:limit])
    except Exception:
        raw = []
    rows = []
    for comment in raw:
        if getattr(comment, "body", None) is None:
            continue
        author = comment.author.name if getattr(comment, "author", None) else "[deleted]"
        score = getattr(comment, "score", "")
        rows.append(
            {
                "author": author,
                "body": str(comment.body or ""),
                "score": "" if score is None else score,
            }
        )
        if len(rows) >= limit:
            break
    while len(rows) < limit:
        rows.append(_empty_comment())
    return rows


def _flatten_comments(comments: list[dict]) -> dict:
    flat = {}
    for i, comment in enumerate(comments, start=1):
        flat[f"comment-body-{i}"] = comment.get("body", "")
        flat[f"comment-body-score-{i}"] = comment.get("score", "")
        flat[f"comment-author-{i}"] = comment.get("author", "")
    return flat


def scrape_cache_path(subreddit: str, listing: str, time_filter: str = "all") -> Path:
    name = normalize_subreddit(subreddit)
    listing = listing.lower().strip()
    window = (time_filter or "all") if listing == "top" else "na"
    return DATA_DIR / f"r_{name}_{listing}_{window}_corpus.csv"


def raw_filename(subreddit: str, n_posts: int) -> str:
    name = normalize_subreddit(subreddit)
    return f"{name}_{int(n_posts)}_raw.csv"


def raw_path(subreddit: str, n_posts: int) -> Path:
    return DATA_DIR / raw_filename(subreddit, n_posts)


def list_raw_files(subreddit: str | None = None) -> list[Path]:
    ensure_dirs()
    if subreddit:
        try:
            name = normalize_subreddit(subreddit)
        except ValueError:
            return []
        paths = DATA_DIR.glob(f"{name}_*_raw.csv")
    else:
        paths = DATA_DIR.glob("*_raw.csv")
    return sorted(paths, key=lambda path: path.stat().st_mtime, reverse=True)


def load_raw_csv(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    frame = pd.read_csv(path)
    if frame.empty or not CACHE_REQUIRED_COLS.issubset(frame.columns):
        return None
    if "post_id" in frame.columns:
        frame = frame.drop_duplicates(subset=["post_id"], keep="first")
    return frame.reset_index(drop=True)


def load_cached_posts(
    subreddit: str,
    listing: str = "hot",
    time_filter: str = "all",
) -> pd.DataFrame | None:
    name = normalize_subreddit(subreddit)
    raw_files = list_raw_files(name)
    if raw_files:
        loaded = load_raw_csv(raw_files[0])
        if loaded is not None:
            return loaded
    path = scrape_cache_path(name, listing, time_filter)
    return load_raw_csv(path)


def save_cached_posts(
    frame: pd.DataFrame,
    subreddit: str,
    listing: str,
    time_filter: str = "all",
    overwrite: bool = False,
) -> Path:
    ensure_dirs()
    path = scrape_cache_path(subreddit, listing, time_filter)
    if not overwrite and path.is_file():
        existing = pd.read_csv(path)
        if not existing.empty and len(existing) >= len(frame):
            return path
    frame.to_csv(path, index=False)
    return path


def save_raw_posts(frame: pd.DataFrame, subreddit: str) -> Path:
    ensure_dirs()
    path = raw_path(subreddit, len(frame))
    frame.to_csv(path, index=False)
    return path


def _fetch_listing_batch(sub, listing: str, time_filter: str, limit: int, after: str | None):
    kwargs = {"limit": limit, "params": {"after": after}}
    if listing == "hot":
        return list(sub.hot(**kwargs))
    if listing == "new":
        return list(sub.new(**kwargs))
    return list(sub.top(time_filter=time_filter or "all", **kwargs))


def _row_from_submission(
    submission,
    name: str,
    listing: str,
    include_stickied: bool = False,
    require_selftext: bool = True,
    comment_sort: str = "top",
    comments_limit: int = TOP_COMMENTS,
) -> dict | None:
    if not include_stickied and getattr(submission, "stickied", False):
        return None
    post_id = str(submission.id)
    selftext = submission.selftext or ""
    stripped = str(selftext).strip()
    if stripped in {"[removed]", "[deleted]"}:
        return None
    if require_selftext and not stripped:
        return None
    if getattr(submission, "removed_by_category", None):
        return None

    title = submission.title or ""
    n_words = word_count(selftext)
    extra = {}
    if comments_limit > 0:
        comments = top_comments(submission, comment_sort=comment_sort, limit=comments_limit)
        extra = _flatten_comments(comments)
    created = datetime.fromtimestamp(
        float(submission.created_utc), tz=timezone.utc
    ).strftime("%Y-%m-%d %H:%M UTC")
    author = (
        submission.author.name
        if getattr(submission, "author", None)
        else "[deleted]"
    )
    permalink = f"https://www.reddit.com{submission.permalink}"
    return {
        "post_id": post_id,
        "title": title,
        "post_author": author,
        "selftext": selftext,
        "score": int(submission.score or 0),
        "ups": int(submission.ups or 0),
        "downs": int(submission.downs or 0),
        "url": submission.url or permalink,
        "permalink": permalink,
        "created_utc": created,
        "post_word_count": n_words,
        "post_size_category": size_category(n_words),
        **extra,
        "data_source": f"R-{name}",
        "listing": listing,
    }


def scrape_and_save(
    subreddit: str,
    listing: str = "hot",
    time_filter: str = "all",
    progress=None,
    use_cache: bool = True,
    raw_file: Path | None = None,
    total_posts: int = TOTAL_POSTS,
    batch_size: int = BATCH_SIZE,
    rate_limit_sleep_s: int = RATE_LIMIT_SLEEP_S,
    include_stickied: bool = False,
    require_selftext: bool = True,
    comment_sort: str = "top",
    comments_limit: int = TOP_COMMENTS,
) -> dict:
    listing = listing.lower().strip()
    if listing not in {"hot", "top", "new"}:
        raise ValueError("Listing must be hot, top, or new.")
    total_posts = int(max(1, min(int(total_posts), TOTAL_POSTS)))
    batch_size = int(max(1, min(int(batch_size), total_posts)))
    rate_limit_sleep_s = int(max(0, rate_limit_sleep_s))
    comments_limit = int(max(0, min(int(comments_limit), TOP_COMMENTS)))
    comment_sort = comment_sort if comment_sort in {"top", "best", "new"} else "top"

    name = normalize_subreddit(subreddit)
    from_cache = False
    frame = None
    loaded_path = None
    if use_cache:
        if raw_file is not None:
            frame = load_raw_csv(Path(raw_file))
            loaded_path = Path(raw_file)
        else:
            existing = list_raw_files(name)
            if existing:
                loaded_path = existing[0]
                frame = load_raw_csv(loaded_path)
        if frame is not None:
            from_cache = True
            if progress:
                progress(f"Using {len(frame)} saved posts from data/{loaded_path.name}")
            return _scrape_result(
                frame,
                loaded_path,
                name,
                listing,
                time_filter,
                total_posts,
                from_cache=True,
            )

    reddit = _reddit_client()
    batch_count = max(1, (total_posts + batch_size - 1) // batch_size)
    if progress:
        progress(
            f"Opening r/{name} (up to {total_posts} posts in batches of {batch_size}"
            + (f", {rate_limit_sleep_s}s pause between batches)…" if rate_limit_sleep_s else ")…")
        )

    try:
        sub = reddit.subreddit(name)
        _ = sub.id
    except Exception as exc:
        detail = str(exc).strip() or exc.__class__.__name__
        raise RuntimeError(
            f"Could not open r/{name}. Check the name, that it is public, and your credentials. ({detail})"
        ) from exc

    rows = []
    seen = set()
    after = None
    for batch_i in range(0, total_posts, batch_size):
        batch_n = batch_i // batch_size + 1
        submissions = _fetch_listing_batch(
            sub, listing, time_filter, batch_size, after
        )
        if not submissions:
            break
        for submission in submissions:
            post_id = str(getattr(submission, "id", ""))
            if not post_id or post_id in seen:
                continue
            row = _row_from_submission(
                submission,
                name,
                listing,
                include_stickied=include_stickied,
                require_selftext=require_selftext,
                comment_sort=comment_sort,
                comments_limit=comments_limit,
            )
            if row is None:
                continue
            seen.add(post_id)
            rows.append(row)
        if progress:
            progress(
                f"Fetched {len(submissions)} posts (batch {batch_n}/{batch_count}). "
                f"Saved so far: {len(rows)}."
            )
        after = submissions[-1].fullname
        if len(submissions) < batch_size or batch_n >= batch_count:
            break
        if rate_limit_sleep_s:
            if progress:
                progress(
                    f"Fetched {batch_size} posts. Waiting {rate_limit_sleep_s}s to avoid rate limit…"
                )
            time.sleep(rate_limit_sleep_s)

    if not rows:
        raise RuntimeError(
            f"No usable posts returned from r/{name}. "
            "Try another listing, or a public subreddit with text posts."
        )
    frame = pd.DataFrame(rows)
    saved = save_raw_posts(frame, name)
    if progress:
        progress(f"Collected {len(frame)} posts. Saved data/{saved.name}")
    return _scrape_result(
        frame,
        saved,
        name,
        listing,
        time_filter,
        total_posts,
        from_cache=False,
    )


def _scrape_result(
    frame: pd.DataFrame,
    path: Path,
    name: str,
    listing: str,
    time_filter: str,
    total_posts: int,
    from_cache: bool,
) -> dict:
    return {
        "posts": frame,
        "n": len(frame),
        "n_requested": total_posts,
        "path": path,
        "filename": path.name,
        "relpath": f"data/{path.name}",
        "subreddit": name,
        "listing": listing,
        "time_filter": time_filter if listing == "top" else "",
        "from_cache": from_cache,
    }


def run_scrape(*args, **kwargs) -> dict:
    """UI entry point for scrape-and-save. Same as scrape_and_save."""
    return scrape_and_save(*args, **kwargs)


def scrape_posts(
    subreddit: str,
    listing: str = "hot",
    time_filter: str = "all",
    progress=None,
    use_cache: bool = True,
    **kwargs,
) -> pd.DataFrame:
    result = scrape_and_save(
        subreddit,
        listing=listing,
        time_filter=time_filter,
        progress=progress,
        use_cache=use_cache,
        **kwargs,
    )
    return result["posts"]
