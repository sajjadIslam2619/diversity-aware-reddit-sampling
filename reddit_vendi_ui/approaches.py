"""Approach 1 (embed, cluster, similarity) and Approach 2 (mental, emotion, size)."""

from __future__ import annotations

import gc
import os
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from vendi import category_kernel, cosine_kernel, mean_off_diagonal, vendi_score

EMBED_MODEL = "sentence-transformers/all-roberta-large-v1"
MENTAL_MODEL = "SajjadIslam/multiMentalRoBERTA-6-class"
EMOTION_MODEL = "tasinhoque/roberta-large-go-emotions"

Progress = Callable[[str], None]


def _noop(_message: str) -> None:
    return None


def _token() -> str | None:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    return token or None


def _release(*objs) -> None:
    for obj in objs:
        del obj
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return


def _classify(texts: list[str], model_name: str, progress: Progress, label: str) -> list[str]:
    import torch
    from transformers import pipeline

    progress(f"Loading {label} model (first run downloads it)…")
    device = 0 if torch.cuda.is_available() else -1
    kwargs = {
        "model": model_name,
        "tokenizer": model_name,
        "truncation": True,
        "max_length": 512,
        "device": device,
    }
    token = _token()
    if token:
        kwargs["token"] = token
    try:
        clf = pipeline("text-classification", **kwargs)
    except TypeError:
        kwargs.pop("token", None)
        clf = pipeline("text-classification", **kwargs)
    except Exception:
        if device == 0:
            kwargs["device"] = -1
            clf = pipeline("text-classification", **kwargs)
        else:
            raise

    preds: list[str] = []
    batch_size = 8
    try:
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            try:
                out = clf(batch, batch_size=batch_size)
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower() or kwargs.get("device") == -1:
                    raise
                _release(clf)
                kwargs["device"] = -1
                clf = pipeline("text-classification", **kwargs)
                out = clf(batch, batch_size=batch_size)
            preds.extend(item["label"] for item in out)
            progress(f"{label}: {min(start + batch_size, len(texts))} / {len(texts)}")
    finally:
        _release(clf)
    return preds


def run_approach_1(frame: pd.DataFrame, k: int, progress: Progress | None = None) -> dict:
    """Embed posts, cluster, then cosine similarity and Vendi on that kernel."""
    progress = progress or _noop
    texts = frame["post_text"].fillna("[no text]").astype(str).tolist()
    n = len(texts)
    if n < 2:
        raise RuntimeError("Approach 1 needs at least 2 posts.")

    progress(f"Loading embedding model (first run downloads {EMBED_MODEL})…")
    from sentence_transformers import SentenceTransformer

    token = _token()
    try:
        model = SentenceTransformer(EMBED_MODEL, token=token) if token else SentenceTransformer(EMBED_MODEL)
    except TypeError:
        model = SentenceTransformer(EMBED_MODEL)

    chunks = []
    batch_size = 8
    try:
        for start in range(0, n, batch_size):
            chunk = model.encode(
                texts[start : start + batch_size],
                show_progress_bar=False,
                convert_to_numpy=True,
                batch_size=batch_size,
            )
            chunks.append(np.asarray(chunk))
            progress(f"Embedding posts: {min(start + batch_size, n)} / {n}")
        embeddings = np.vstack(chunks)
    finally:
        _release(model)

    rounded = np.round(embeddings, 5)
    n_unique = int(np.unique(rounded, axis=0).shape[0])
    k_used = int(max(1, min(int(k), n, n_unique)))
    if k_used < 2:
        labels = np.zeros(n, dtype=int)
        centers = embeddings.mean(axis=0, keepdims=True)
    else:
        kmeans = KMeans(n_clusters=k_used, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        centers = kmeans.cluster_centers_

    cluster_names = [f"cluster_{i + 1}" for i in range(k_used)]
    assigned = [f"cluster_{i + 1}" for i in labels]
    kernel = cosine_kernel(embeddings)
    center_sim = cosine_similarity(centers)
    center_df = pd.DataFrame(center_sim, index=cluster_names, columns=cluster_names)
    counts = (
        pd.Series(assigned, name="post_count")
        .value_counts()
        .reindex(cluster_names, fill_value=0)
        .rename_axis("cluster")
        .reset_index(name="post_count")
    )

    scored = frame.copy()
    scored["cluster"] = assigned
    return {
        "posts": scored,
        "k_requested": int(k),
        "k_used": k_used,
        "embedding_dim": int(embeddings.shape[1]),
        "vendi": vendi_score(kernel),
        "mean_similarity": mean_off_diagonal(kernel),
        "cluster_counts": counts,
        "centroid_similarity": center_df,
        "model": EMBED_MODEL,
    }


def run_approach_2(frame: pd.DataFrame, progress: Progress | None = None) -> dict:
    """Mental status, emotion, and post size, then a category similarity kernel."""
    progress = progress or _noop
    texts = frame["post_text"].fillna("[no text]").astype(str).tolist()
    if len(texts) < 2:
        raise RuntimeError("Approach 2 needs at least 2 posts.")

    mental = _classify(texts, MENTAL_MODEL, progress, "Mental status")
    emotion = _classify(texts, EMOTION_MODEL, progress, "Emotion")

    scored = frame.copy()
    scored["mental_health_class"] = mental
    scored["post_emotion"] = emotion
    scored["size_category"] = scored["post_size_category"].fillna("out_of_band").astype(str)
    scored["mental_status_category"] = scored["mental_health_class"].fillna("unknown").astype(str)
    scored["emotion_category"] = scored["post_emotion"].fillna("unknown").astype(str)
    scored["size_mental_emotion_label"] = (
        scored["size_category"]
        + " | "
        + scored["mental_status_category"]
        + " | "
        + scored["emotion_category"]
    )

    kernel = category_kernel(
        [
            scored["size_category"].to_numpy(),
            scored["mental_status_category"].to_numpy(),
            scored["emotion_category"].to_numpy(),
        ]
    )
    return {
        "posts": scored,
        "vendi": vendi_score(kernel),
        "mean_similarity": mean_off_diagonal(kernel),
        "mental_model": MENTAL_MODEL,
        "emotion_model": EMOTION_MODEL,
    }
