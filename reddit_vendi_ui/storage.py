"""Local SQLite store for scrape runs. Does not touch existing project files."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DB_PATH = OUTPUT_DIR / "runs.sqlite"


def _connect() -> sqlite3.Connection:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            subreddit TEXT NOT NULL,
            listing TEXT NOT NULL,
            time_filter TEXT,
            n_requested INTEGER NOT NULL,
            n_scraped INTEGER NOT NULL,
            k_requested INTEGER,
            k_used INTEGER,
            vendi_approach1 REAL,
            vendi_approach2 REAL,
            mean_sim_approach1 REAL,
            mean_sim_approach2 REAL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS posts (
            run_id INTEGER NOT NULL,
            post_id TEXT,
            title TEXT,
            selftext TEXT,
            score INTEGER,
            url TEXT,
            created_utc TEXT,
            post_word_count INTEGER,
            post_size_category TEXT,
            mental_health_class TEXT,
            post_emotion TEXT,
            cluster TEXT,
            comment_body TEXT,
            data_source TEXT
        )
        """
    )
    conn.commit()
    return conn


def save_run(summary: dict, posts: pd.DataFrame) -> int:
    conn = _connect()
    try:
        cur = conn.execute(
            """
            INSERT INTO runs (
                created_at, subreddit, listing, time_filter, n_requested, n_scraped,
                k_requested, k_used, vendi_approach1, vendi_approach2,
                mean_sim_approach1, mean_sim_approach2
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now().isoformat(timespec="seconds"),
                summary["subreddit"],
                summary["listing"],
                summary.get("time_filter") or "",
                int(summary["n_requested"]),
                int(summary["n_scraped"]),
                int(summary["k_requested"]),
                int(summary["k_used"]),
                float(summary["vendi_approach1"]),
                float(summary["vendi_approach2"]),
                float(summary["mean_sim_approach1"]),
                float(summary["mean_sim_approach2"]),
            ),
        )
        run_id = int(cur.lastrowid)
        rows = []
        for rec in posts.to_dict(orient="records"):
            rows.append(
                (
                    run_id,
                    rec.get("post_id"),
                    rec.get("title"),
                    rec.get("selftext"),
                    rec.get("score"),
                    rec.get("url"),
                    rec.get("created_utc"),
                    rec.get("post_word_count"),
                    rec.get("post_size_category"),
                    rec.get("mental_health_class"),
                    rec.get("post_emotion"),
                    rec.get("cluster"),
                    rec.get("comment_body"),
                    rec.get("data_source"),
                )
            )
        conn.executemany(
            """
            INSERT INTO posts (
                run_id, post_id, title, selftext, score, url, created_utc,
                post_word_count, post_size_category, mental_health_class,
                post_emotion, cluster, comment_body, data_source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
        return run_id
    finally:
        conn.close()


def list_runs(limit: int = 8) -> pd.DataFrame:
    if not DB_PATH.is_file():
        return pd.DataFrame()
    conn = _connect()
    try:
        return pd.read_sql_query(
            """
            SELECT id, created_at, subreddit, listing, n_scraped,
                   k_used, vendi_approach1, vendi_approach2
            FROM runs
            ORDER BY id DESC
            LIMIT ?
            """,
            conn,
            params=(int(limit),),
        )
    finally:
        conn.close()
