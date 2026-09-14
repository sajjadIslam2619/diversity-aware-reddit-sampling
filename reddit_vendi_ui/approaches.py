"""Approach 1 (embed, cluster, similarity) and Approach 2 (mental, emotion, size)."""

from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from paths import model_path
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


def _is_local_model(path: Path) -> bool:
    if not path.is_dir():
        return False
    has_config = (path / "config.json").is_file() or (path / "modules.json").is_file()
    if not has_config:
        return False
    weight_names = (
        "model.safetensors",
        "pytorch_model.bin",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    )
    if any((path / name).is_file() for name in weight_names):
        return True
    for child in path.rglob("*"):
        if child.is_file() and child.name in weight_names:
            return True
        if child.is_file() and child.suffix in {".bin", ".safetensors"}:
            return True
    return False


def resolve_local_model(repo_id: str, progress: Progress, label: str) -> str:
    dest = model_path(repo_id)
    if _is_local_model(dest):
        progress(f"Using cached {label} model in models/{dest.name}")
        return str(dest)

    progress(f"Downloading {label} model to models/{dest.name}…")
    from huggingface_hub import snapshot_download

    kwargs: dict = {"repo_id": repo_id, "local_dir": str(dest)}
    token = _token()
    if token:
        kwargs["token"] = token
    snapshot_download(**kwargs)
    if not _is_local_model(dest):
        raise RuntimeError(
            f"Downloaded {repo_id} to models/{dest.name}, but the folder looks incomplete."
        )
    return str(dest)


def _classify(texts: list[str], model_name: str, progress: Progress, label: str) -> list[str]:
    import torch
    from transformers import pipeline

    local_model = resolve_local_model(model_name, progress, label)
    device = 0 if torch.cuda.is_available() else -1
    kwargs = {
        "model": local_model,
        "tokenizer": local_model,
        "truncation": True,
        "max_length": 512,
        "device": device,
    }
    try:
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


def _selftexts(frame: pd.DataFrame) -> list[str]:
    return (
        frame["selftext"]
        .fillna("")
        .astype(str)
        .str.strip()
        .replace("", "[no text]")
        .tolist()
    )


def encode_texts(texts: list[str], progress: Progress | None = None) -> np.ndarray:
    progress = progress or _noop
    local_model = resolve_local_model(EMBED_MODEL, progress, "embedding")
    from sentence_transformers import SentenceTransformer

    try:
        model = SentenceTransformer(local_model, local_files_only=True)
    except TypeError:
        model = SentenceTransformer(local_model)

    chunks = []
    batch_size = 8
    n = len(texts)
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
        return np.vstack(chunks)
    finally:
        _release(model)


def corpus_is_labeled(frame: pd.DataFrame) -> bool:
    needed = ("mental_health_class", "post_emotion")
    if any(col not in frame.columns for col in needed):
        return False
    mental = frame["mental_health_class"].fillna("").astype(str).str.strip()
    emotion = frame["post_emotion"].fillna("").astype(str).str.strip()
    return bool((mental != "").all() and (emotion != "").all())


def label_corpus(frame: pd.DataFrame, progress: Progress | None = None) -> pd.DataFrame:
    """Mental status and emotion labels for every post (cached on the corpus CSV)."""
    progress = progress or _noop
    scored = frame.copy()
    if corpus_is_labeled(scored):
        progress("Using saved mental-status and emotion labels.")
    else:
        texts = _selftexts(scored)
        if len(texts) < 2:
            raise RuntimeError("Need at least 2 posts to label and sample.")
        mental = _classify(texts, MENTAL_MODEL, progress, "Mental status")
        emotion = _classify(texts, EMOTION_MODEL, progress, "Emotion")
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
    return scored


def sample_approach_1(
    frame: pd.DataFrame,
    embeddings: np.ndarray,
    k: int,
    progress: Progress | None = None,
) -> dict:
    """k clusters on the full corpus; one post nearest each centroid (k diverse posts)."""
    progress = progress or _noop
    n = len(frame)
    if n < 2:
        raise RuntimeError("Approach 1 needs at least 2 posts.")
    vectors = np.asarray(embeddings, dtype=np.float64)
    if vectors.shape[0] != n:
        raise RuntimeError("Embedding rows do not match the corpus.")

    rounded = np.round(vectors, 5)
    n_unique = int(np.unique(rounded, axis=0).shape[0])
    k_used = int(max(1, min(int(k), n, n_unique)))
    progress(f"Approach 1: clustering corpus into k={k_used}…")
    if k_used < 2:
        labels = np.zeros(n, dtype=int)
        centers = vectors.mean(axis=0, keepdims=True)
    else:
        kmeans = KMeans(n_clusters=k_used, random_state=42, n_init=10)
        labels = kmeans.fit_predict(vectors)
        centers = kmeans.cluster_centers_

    cluster_names = [f"cluster_{i + 1}" for i in range(k_used)]
    assigned = [f"cluster_{i + 1}" for i in labels]
    picked = []
    for c in range(k_used):
        idx = np.where(labels == c)[0]
        dists = np.linalg.norm(vectors[idx] - centers[c], axis=1)
        picked.append(int(idx[int(np.argmin(dists))]))

    scored = frame.copy()
    scored["cluster"] = assigned
    selected = scored.iloc[picked].copy().reset_index(drop=True)
    selected_vectors = vectors[picked]
    center_sim = cosine_similarity(centers)
    center_df = pd.DataFrame(center_sim, index=cluster_names, columns=cluster_names)
    counts = (
        pd.Series(assigned, name="post_count")
        .value_counts()
        .reindex(cluster_names, fill_value=0)
        .rename_axis("cluster")
        .reset_index(name="post_count")
    )
    kernel = cosine_kernel(selected_vectors)
    return {
        "corpus": scored,
        "posts": selected,
        "embeddings": selected_vectors,
        "k_requested": int(k),
        "k_used": k_used,
        "embedding_dim": int(vectors.shape[1]),
        "vendi": vendi_score(kernel),
        "mean_similarity": mean_off_diagonal(kernel),
        "cluster_counts": counts,
        "centroid_similarity": center_df,
        "model": EMBED_MODEL,
    }


def sample_approach_2(
    frame: pd.DataFrame,
    n: int,
    progress: Progress | None = None,
    embeddings: np.ndarray | None = None,
) -> dict:
    """n posts covering as many size × mental × emotion combinations as possible."""
    progress = progress or _noop
    if len(frame) < 2:
        raise RuntimeError("Approach 2 needs at least 2 posts.")
    n_used = int(max(1, min(int(n), len(frame))))
    progress(f"Approach 2: selecting {n_used} posts across size / mental / emotion groups…")

    group_cols = ["size_category", "mental_status_category", "emotion_category"]
    working = frame.reset_index(drop=True)
    buckets: list[list[int]] = []
    for _, group in working.groupby(group_cols, dropna=False, sort=False):
        order = group.sample(frac=1, random_state=42).index.astype(int).tolist()
        buckets.append(order)
    rng = np.random.RandomState(42)
    rng.shuffle(buckets)

    selected_idx: list[int] = []
    round_n = 0
    while len(selected_idx) < n_used:
        progressed = False
        for bucket in buckets:
            if len(selected_idx) >= n_used:
                break
            if round_n < len(bucket):
                selected_idx.append(bucket[round_n])
                progressed = True
        if not progressed:
            break
        round_n += 1

    selected = working.iloc[selected_idx].copy().reset_index(drop=True)
    if embeddings is None:
        kernel = category_kernel(
            [
                selected["size_category"].to_numpy(),
                selected["mental_status_category"].to_numpy(),
                selected["emotion_category"].to_numpy(),
            ]
        )
        selected_vectors = None
    else:
        selected_vectors = np.asarray(embeddings, dtype=np.float64)[selected_idx]
        kernel = cosine_kernel(selected_vectors)

    return {
        "posts": selected,
        "embeddings": selected_vectors,
        "n_requested": int(n),
        "n_selected": len(selected),
        "n_label_groups": len(buckets),
        "vendi": vendi_score(kernel),
        "mean_similarity": mean_off_diagonal(kernel),
        "mental_model": MENTAL_MODEL,
        "emotion_model": EMOTION_MODEL,
    }
