"""Scrape a subreddit with PRAW. Reads credentials from the repo .env."""

from __future__ import annotations

import os
import re
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
CACHE_REQUIRED_COLS = {
    "title",
    "post_author",
    "selftext",
    "score",
    "ups",
    "downs",
    "url",
    "comment-body-1",
    "comment-body-2",
    "comment-body-3",
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


def load_cached_posts(
    subreddit: str,
    listing: str = "hot",
    time_filter: str = "all",
) -> pd.DataFrame | None:
    path = scrape_cache_path(subreddit, listing, time_filter)
    if not path.is_file():
        return None
    frame = pd.read_csv(path)
    if frame.empty or not CACHE_REQUIRED_COLS.issubset(frame.columns):
        return None
    if "post_id" in frame.columns:
        frame = frame.drop_duplicates(subset=["post_id"], keep="first")
    return frame.reset_index(drop=True)


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


def scrape_posts(
    subreddit: str,
    listing: str = "hot",
    time_filter: str = "all",
    progress=None,
    use_cache: bool = True,
) -> pd.DataFrame:
    listing = listing.lower().strip()
    if listing not in {"hot", "top", "new"}:
        raise ValueError("Listing must be hot, top, or new.")

    name = normalize_subreddit(subreddit)
    cache_file = scrape_cache_path(name, listing, time_filter)
    if use_cache:
        cached = load_cached_posts(name, listing, time_filter)
        if cached is not None:
            if progress:
                progress(
                    f"Using {len(cached)} cached posts from data/{cache_file.name}"
                )
            return cached
        if cache_file.is_file() and progress:
            progress(
                f"Cached file data/{cache_file.name} is missing fields; scraping r/{name}…"
            )

    reddit = _reddit_client()
    if progress:
        progress(f"Opening r/{name} (full listing, Reddit typically caps at ~1000)…")

    try:
        sub = reddit.subreddit(name)
        # Force a lookup so a bad name fails before the listing loop.
        _ = sub.id
    except Exception as exc:
        detail = str(exc).strip() or exc.__class__.__name__
        raise RuntimeError(
            f"Could not open r/{name}. Check the name, that it is public, and your credentials. ({detail})"
        ) from exc

    if listing == "hot":
        generator = sub.hot(limit=None)
    elif listing == "new":
        generator = sub.new(limit=None)
    else:
        generator = sub.top(limit=None, time_filter=time_filter or "all")

    rows = []
    seen = set()
    for submission in generator:
        if getattr(submission, "stickied", False):
            continue
        post_id = str(submission.id)
        if post_id in seen:
            continue
        selftext = submission.selftext or ""
        if not str(selftext).strip() or str(selftext).strip() in {"[removed]", "[deleted]"}:
            continue
        if getattr(submission, "removed_by_category", None):
            continue

        title = submission.title or ""
        n_words = word_count(selftext)
        comments = top_comments(submission, comment_sort="top", limit=TOP_COMMENTS)
        created = datetime.fromtimestamp(
            float(submission.created_utc), tz=timezone.utc
        ).strftime("%Y-%m-%d %H:%M UTC")
        author = (
            submission.author.name
            if getattr(submission, "author", None)
            else "[deleted]"
        )
        permalink = f"https://www.reddit.com{submission.permalink}"
        seen.add(post_id)
        rows.append(
            {
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
                **_flatten_comments(comments),
                "data_source": f"R-{name}",
                "listing": listing,
            }
        )
        if progress and len(rows) % 25 == 0:
            progress(f"Scraped {len(rows)} posts from r/{name}…")

    if not rows:
        raise RuntimeError(
            f"No usable posts returned from r/{name}. "
            "Try another listing, or a public subreddit with text posts."
        )
    frame = pd.DataFrame(rows)
    saved = save_cached_posts(
        frame, name, listing, time_filter, overwrite=not use_cache
    )
    if progress:
        progress(f"Saved {len(frame)} posts to data/{saved.name}")
    return frame
