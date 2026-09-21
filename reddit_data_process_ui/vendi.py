"""Vendi score: effective number of dissimilar items.

Friedman & Dieng, "The Vendi Score: A Diversity Evaluation Metric for
Machine Learning" (2022). For a similarity kernel with unit diagonal,
the score is the exponential of the Shannon entropy of the eigenvalues
of K / n. It lies in [1, n]: 1 means every item looks the same, n means
they are mutually dissimilar.
"""

from __future__ import annotations

import numpy as np


def vendi_score(kernel: np.ndarray) -> float:
    k = np.asarray(kernel, dtype=np.float64)
    if k.ndim != 2 or k.shape[0] != k.shape[1]:
        raise ValueError("Kernel must be a square matrix.")
    n = k.shape[0]
    if n == 0:
        return 0.0
    if n == 1:
        return 1.0

    k = 0.5 * (k + k.T)
    evals = np.linalg.eigvalsh(k / n)
    evals = np.clip(evals, 0.0, None)
    total = float(evals.sum())
    if total <= 0.0:
        return 1.0
    probs = evals / total
    probs = probs[probs > 1e-12]
    entropy = float(-np.sum(probs * np.log(probs)))
    return float(np.exp(entropy))


def mean_off_diagonal(kernel: np.ndarray) -> float:
    k = np.asarray(kernel, dtype=np.float64)
    n = k.shape[0]
    if n < 2:
        return float("nan")
    return float((k.sum() - np.trace(k)) / (n * (n - 1)))


def cosine_kernel(embeddings: np.ndarray) -> np.ndarray:
    x = np.asarray(embeddings, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("Embeddings must be a 2-D array.")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    x = x / norms
    kernel = x @ x.T
    np.fill_diagonal(kernel, 1.0)
    return np.clip(kernel, -1.0, 1.0)


def category_kernel(labels: list[np.ndarray]) -> np.ndarray:
    """Average of exact-match kernels. PSD when each match kernel is."""
    if not labels:
        raise ValueError("Need at least one label column.")
    n = len(labels[0])
    kernel = np.zeros((n, n), dtype=np.float64)
    for vals in labels:
        col = np.asarray(vals)
        if len(col) != n:
            raise ValueError("Label columns must have the same length.")
        kernel += (col[:, None] == col[None, :]).astype(np.float64)
    kernel /= float(len(labels))
    np.fill_diagonal(kernel, 1.0)
    return kernel
