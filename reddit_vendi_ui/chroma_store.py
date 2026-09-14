"""Persistent ChromaDB store for post selftext embeddings."""

from __future__ import annotations

import re
from typing import Callable

import numpy as np
import pandas as pd

from paths import CHROMA_DIR, ensure_dirs

Progress = Callable[[str], None]
EncodeFn = Callable[[list[str], Progress], np.ndarray]


def _collection_name(subreddit: str, listing: str) -> str:
    raw = f"r_{subreddit}_{listing}_selftext"
    cleaned = re.sub(r"[^a-zA-Z0-9._-]", "_", raw)
    if len(cleaned) < 3:
        cleaned = f"col_{cleaned}"
    return cleaned[:63]


def _client():
    import chromadb

    ensure_dirs()
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def _collection(subreddit: str, listing: str):
    client = _client()
    return client.get_or_create_collection(
        name=_collection_name(subreddit, listing),
        metadata={"hnsw:space": "cosine"},
    )


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
    collection = _collection(subreddit, listing)
    count = collection.count()
    existing = set()
    if count:
        existing = set(collection.get(limit=count).get("ids") or [])
    missing_idx = [i for i, post_id in enumerate(ids) if post_id not in existing]

    if missing_idx:
        progress(
            f"Embedding {len(missing_idx)} new posts "
            f"({len(ids) - len(missing_idx)} already in ChromaDB)…"
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
        progress("Using embeddings already stored in ChromaDB.")

    fetched = collection.get(ids=ids, include=["embeddings"])
    by_id = {
        post_id: np.asarray(emb, dtype=np.float64)
        for post_id, emb in zip(fetched.get("ids") or [], fetched.get("embeddings") or [])
    }
    missing_after = [i for i, post_id in enumerate(ids) if post_id not in by_id]
    if missing_after:
        raise RuntimeError(
            f"ChromaDB is missing {len(missing_after)} embeddings after upsert."
        )
    return np.vstack([by_id[post_id] for post_id in ids])
