"""Scrape a subreddit with PRAW. Reads credentials from the repo .env."""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
SIZE_BANDS = (
    ("small", 10, 50),
    ("medium", 51, 200),
    ("large", 201, 500),
)


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


def post_text(title: str, selftext: str) -> str:
    text = f"{str(title or '').strip()} {str(selftext or '').strip()}".strip()
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


def _best_comment(submission) -> tuple[str, str]:
    try:
        submission.comment_sort = "best"
        submission.comments.replace_more(limit=0)
        comments = list(submission.comments)[:1]
    except Exception:
        return "", ""
    if not comments:
        return "", ""
    comment = comments[0]
    body = getattr(comment, "body", "") or ""
    score = getattr(comment, "score", "")
    return str(body), "" if score == "" else str(score)


def scrape_posts(
    subreddit: str,
    limit: int,
    listing: str = "hot",
    time_filter: str = "all",
    progress=None,
) -> pd.DataFrame:
    if limit not in (20, 50, 100):
        raise ValueError("Select 20, 50, or 100 posts.")
    listing = listing.lower().strip()
    if listing not in {"hot", "top", "new"}:
        raise ValueError("Listing must be hot, top, or new.")

    name = normalize_subreddit(subreddit)
    reddit = _reddit_client()
    if progress:
        progress(f"Opening r/{name}…")

    try:
        sub = reddit.subreddit(name)
        # Force a lookup so a bad name fails before the listing loop.
        _ = sub.id
    except Exception as exc:
        detail = str(exc).strip() or exc.__class__.__name__
        raise RuntimeError(
            f"Could not open r/{name}. Check the name, that it is public, and your credentials. ({detail})"
        ) from exc

    raw_limit = min(limit + 25, 1000)
    if listing == "hot":
        generator = sub.hot(limit=raw_limit)
    elif listing == "new":
        generator = sub.new(limit=raw_limit)
    else:
        generator = sub.top(limit=raw_limit, time_filter=time_filter or "all")

    rows = []
    for submission in generator:
        if getattr(submission, "stickied", False):
            continue
        selftext = submission.selftext or ""
        if selftext.strip() in {"[removed]", "[deleted]"}:
            continue
        if getattr(submission, "removed_by_category", None):
            continue

        title = submission.title or ""
        text = post_text(title, selftext)
        n_words = word_count(text if text != "[no text]" else "")
        comment_body, comment_score = _best_comment(submission)
        created = datetime.fromtimestamp(
            float(submission.created_utc), tz=timezone.utc
        ).strftime("%Y-%m-%d %H:%M UTC")
        rows.append(
            {
                "post_id": submission.id,
                "title": title,
                "selftext": selftext,
                "post_text": text,
                "score": int(submission.score or 0),
                "url": f"https://www.reddit.com{submission.permalink}",
                "created_utc": created,
                "post_word_count": n_words,
                "post_size_category": size_category(n_words),
                "comment_body": comment_body,
                "comment_score": comment_score,
                "data_source": f"R-{name}",
                "listing": listing,
            }
        )
        if progress and len(rows) % 5 == 0:
            progress(f"Scraped {len(rows)} / {limit} posts from r/{name}…")
        if len(rows) >= limit:
            break

    if not rows:
        raise RuntimeError(
            f"No usable posts returned from r/{name}. "
            "Try another listing, or a public subreddit with text posts."
        )
    return pd.DataFrame(rows)
