"""只按未来相似度挑正例。不看过去，也不造负例。"""

from __future__ import annotations

import numpy as np

from retriever.data import WindowSet
from retriever.sampling import window_similarity


def pick_positives(future_similarity: np.ndarray, quantile: float, min_similarity: float) -> np.ndarray:
    floor = max(float(np.quantile(future_similarity, quantile)), float(min_similarity))
    positive = np.flatnonzero(future_similarity >= floor)
    if len(positive) == 0:
        positive = np.array([int(np.argmax(future_similarity))], dtype=np.int64)
    return positive.astype(np.int64)


def build_positives(
    queries: WindowSet,
    database: WindowSet,
    temperature: float,
    quantile: float,
    min_similarity: float,
) -> list[np.ndarray]:
    future_similarity = window_similarity(queries.y, database.y, temperature)
    return [pick_positives(future_similarity[row], quantile, min_similarity) for row in range(len(queries))]


def positive_report(pools: list[np.ndarray]) -> dict:
    sizes = [len(group) for group in pools]
    return {"mean_positives": round(float(np.mean(sizes)) if sizes else 0.0, 1)}
