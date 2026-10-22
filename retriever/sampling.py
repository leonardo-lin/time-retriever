"""用未来轨迹给历史候选打标签。标签不依赖检索器参数，所以训练前算一次。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from retriever.config import Config
from retriever.data import WindowSet


@dataclass
class Pools:
    positive: list[np.ndarray]
    hard: list[np.ndarray]
    easy: list[np.ndarray]


def zscore_rows(windows: np.ndarray) -> np.ndarray:
    values = np.ascontiguousarray(windows, dtype=np.float32)
    flat = values.reshape(-1, values.shape[-1])
    centered = flat - flat.mean(axis=1, keepdims=True)
    scale = flat.std(axis=1, keepdims=True)
    np.maximum(scale, 1e-6, out=scale)
    return (centered / scale).reshape(values.shape)


def window_similarity(left: np.ndarray, right: np.ndarray, temperature: float) -> np.ndarray:
    """每条 left 对每条 right 的相似度，形状 (len(left), len(right))。

    距离 = 0.5 * 标准化 MSE + 0.5 * (1 - 相关系数)。
    窗口已经按自身标准化后，MSE 和相关系数是同一个形状差异的两种写法，
    这里按规格把两者合在一起。
    """
    left_z = zscore_rows(left)
    right_z = zscore_rows(right)
    correlation = left_z @ right_z.T / left_z.shape[-1]
    left_energy = np.mean(left_z * left_z, axis=1, keepdims=True)
    right_energy = np.mean(right_z * right_z, axis=1)
    mse = left_energy + right_energy - 2.0 * correlation
    distance = 0.5 * mse + 0.5 * (1.0 - correlation)
    return np.exp(-distance / temperature).astype(np.float32)


def classify_candidates(past_similarity: np.ndarray, future_similarity: np.ndarray, config: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    future_cut = float(np.quantile(future_similarity, config.positive_future_quantile))
    positive_floor = max(future_cut, float(config.positive_min_similarity))
    positive = np.flatnonzero(future_similarity >= positive_floor)
    low_future = np.flatnonzero(future_similarity <= np.quantile(future_similarity, config.low_future_quantile))
    high_past = np.flatnonzero(past_similarity >= np.quantile(past_similarity, config.high_past_quantile))
    low_past = np.flatnonzero(past_similarity <= np.quantile(past_similarity, config.low_past_quantile))
    hard = np.intersect1d(high_past, low_future, assume_unique=False)
    easy = np.intersect1d(low_past, low_future, assume_unique=False)
    if len(positive) == 0:
        positive = np.array([int(np.argmax(future_similarity))], dtype=np.int64)
    return positive.astype(np.int64), hard.astype(np.int64), easy.astype(np.int64)


def build_pools(queries: WindowSet, database: WindowSet, config: Config) -> Pools:
    future_similarity = window_similarity(queries.y, database.y, config.future_temperature)
    past_similarity = window_similarity(queries.x, database.x, config.future_temperature)
    positive: list[np.ndarray] = []
    hard: list[np.ndarray] = []
    easy: list[np.ndarray] = []
    for row in range(len(queries)):
        pos, hard_idx, easy_idx = classify_candidates(past_similarity[row], future_similarity[row], config)
        positive.append(pos)
        hard.append(hard_idx)
        easy.append(easy_idx)
    return Pools(positive, hard, easy)


def pool_report(pools: Pools) -> dict:
    def mean_size(groups: list[np.ndarray]) -> float:
        if not groups:
            return 0.0
        return float(np.mean([len(group) for group in groups]))

    return {
        "queries": len(pools.positive),
        "mean_positives": round(mean_size(pools.positive), 1),
        "mean_hard_negatives": round(mean_size(pools.hard), 1),
        "mean_easy_negatives": round(mean_size(pools.easy), 1),
        "queries_with_hard_negatives": int(sum(len(group) > 0 for group in pools.hard)),
    }
