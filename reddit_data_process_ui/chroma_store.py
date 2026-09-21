"""Persistent ChromaDB store for post selftext embeddings."""

from __future__ import annotations

import re
from typing import Callable

import numpy as np
import pandas as pd

from paths import CHROMA_DIR, ensure_dirs

Progress = Callable[[str], None]
EncodeFn = Callable[[list[str], Progress], np.ndarray]


def _collection_name(subreddit: str) -> str:
    raw = f"r_{subreddit}_selftext"
    cleaned = re.sub(r"[^a-zA-Z0-9._-]", "_", raw)
    if len(cleaned) < 3:
        cleaned = f"col_{cleaned}"
    return cleaned[:63]


def _client():
    import chromadb

    ensure_dirs()
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def _collection(subreddit: str):
    client = _client()
    return client.get_or_create_collection(
        name=_collection_name(subreddit),
        metadata={"hnsw:space": "cosine", "subreddit": subreddit},
    )


def stored_ids(subreddit: str) -> set[str]:
    collection = _collection(subreddit)
    count = collection.count()
    if not count:
        return set()
    return set(_as_list(collection.get(limit=count).get("ids")))


def embedding_coverage(frame: pd.DataFrame, subreddit: str) -> dict:
    """How many corpus posts already have embeddings for this subreddit."""
    if "post_id" not in frame.columns:
        return {
            "subreddit": subreddit,
            "corpus": len(frame),
            "stored": 0,
            "missing": len(frame),
            "collection_total": 0,
        }
    ids = frame["post_id"].astype(str).tolist()
    existing = stored_ids(subreddit)
    stored = sum(1 for post_id in ids if post_id in existing)
    return {
        "subreddit": subreddit,
        "corpus": len(ids),
        "stored": stored,
        "missing": len(ids) - stored,
        "collection_total": len(existing),
    }


def _meta_scalar(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and pd.isna(value):
            return ""
        return value
    return str(value)[:500]


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    return list(value)


def embeddings_for_posts(
    frame: pd.DataFrame,
    encode: EncodeFn,
    subreddit: str,
    listing: str,
    embed_model: str,
    progress: Progress | None = None,
) -> np.ndarray:
    """Return selftext embeddings in frame order, encoding only posts missing from Chroma."""
    progress = progress or (lambda _msg: None)
    if "post_id" not in frame.columns:
        raise RuntimeError("Corpus needs a post_id column for Chroma.")
    ids = frame["post_id"].astype(str).tolist()
    collection = _collection(subreddit)
    existing = stored_ids(subreddit)
    missing_idx = [i for i, post_id in enumerate(ids) if post_id not in existing]
    progress(
        f"Checked r/{subreddit} in ChromaDB: "
        f"{len(ids) - len(missing_idx)} / {len(ids)} posts already embedded."
    )

    if missing_idx:
        progress(
            f"Embedding {len(missing_idx)} new posts for r/{subreddit} "
            f"and saving them to ChromaDB…"
        )
        texts = (
            frame["selftext"]
            .iloc[missing_idx]
            .fillna("")
            .astype(str)
            .str.strip()
            .replace("", "[no text]")
            .tolist()
        )
        vectors = np.asarray(encode(texts, progress), dtype=np.float32)
        batch = 64
        for start in range(0, len(missing_idx), batch):
            chunk = missing_idx[start : start + batch]
            local = list(range(start, min(start + batch, len(missing_idx))))
            metas = []
            for i in chunk:
                row = frame.iloc[i]
                metas.append(
                    {
                        "post_id": ids[i],
                        "title": _meta_scalar(row.get("title")),
                        "subreddit": subreddit,
                        "listing": listing,
                        "embed_model": embed_model,
                        "embedded_field": "selftext",
                    }
                )
            collection.upsert(
                ids=[ids[i] for i in chunk],
                embeddings=vectors[local].tolist(),
                documents=frame["selftext"].iloc[chunk].fillna("").astype(str).tolist(),
                metadatas=metas,
            )
    else:
        progress(f"Using embeddings already stored in ChromaDB for r/{subreddit}.")

    fetched = collection.get(ids=ids, include=["embeddings"])
    fetched_ids = _as_list(fetched.get("ids"))
    fetched_embs = fetched.get("embeddings")
    if fetched_embs is None:
        fetched_embs = []
    by_id = {
        post_id: np.asarray(emb, dtype=np.float64)
        for post_id, emb in zip(fetched_ids, fetched_embs)
    }
    missing_after = [i for i, post_id in enumerate(ids) if post_id not in by_id]
    if missing_after:
        raise RuntimeError(
            f"ChromaDB is missing {len(missing_after)} embeddings after upsert."
        )
    return np.vstack([by_id[post_id] for post_id in ids])
