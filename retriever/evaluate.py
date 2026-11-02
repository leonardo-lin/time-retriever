"""训练结束后，检查检索器找到的案例未来是否更相似。"""

from __future__ import annotations

import numpy as np
import torch

from retriever.config import Config
from retriever.data import WindowSet
from retriever.model import RetrieverEncoder
from retriever.sampling import classify_candidates, window_similarity


def embed(model: RetrieverEncoder, windows: np.ndarray, device: torch.device, batch_size: int = 8) -> torch.Tensor:
    model.eval()
    pieces: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            batch = torch.from_numpy(np.ascontiguousarray(windows[start : start + batch_size])).to(device)
            pieces.append(model(batch).cpu())
    return torch.cat(pieces, dim=0)


def _per_query_mean(similarity: np.ndarray, index: np.ndarray) -> np.ndarray:
    rows = np.arange(index.shape[0])[:, None]
    return similarity[rows, index].mean(axis=1)


def _hard_sets(past_similarity: np.ndarray, future_similarity: np.ndarray, config: Config) -> list[set[int]]:
    sets: list[set[int]] = []
    for row in range(len(past_similarity)):
        _, hard, _ = classify_candidates(past_similarity[row], future_similarity[row], config)
        sets.append(set(hard.tolist()))
    return sets


def _hard_rates(index: np.ndarray, hard_sets: list[set[int]], top_k: int) -> tuple[float | None, list[float | None]]:
    hits: list[float] = []
    per_query: list[float | None] = []
    for row, chosen in enumerate(index):
        if not hard_sets[row]:
            per_query.append(None)
            continue
        rate = sum(int(item in hard_sets[row]) for item in chosen) / top_k
        hits.append(rate)
        per_query.append(rate)
    return (None if not hits else float(np.mean(hits)), per_query)


def score_retrieval(
    retrieved: np.ndarray,
    oracle: np.ndarray,
    random_index: np.ndarray,
    future_similarity: np.ndarray,
    past_similarity: np.ndarray,
    config: Config,
) -> dict:
    retrieved_per = _per_query_mean(future_similarity, retrieved)
    oracle_per = _per_query_mean(future_similarity, oracle)
    random_per = _per_query_mean(future_similarity, random_index)
    retrieved_score = float(retrieved_per.mean())
    oracle_score = float(oracle_per.mean())
    random_score = float(random_per.mean())

    hard_sets = _hard_sets(past_similarity, future_similarity, config)
    retrieved_hnr, retrieved_hnr_per = _hard_rates(retrieved, hard_sets, config.top_k)
    oracle_hnr, oracle_hnr_per = _hard_rates(oracle, hard_sets, config.top_k)
    random_hnr, random_hnr_per = _hard_rates(random_index, hard_sets, config.top_k)

    denom = oracle_score - random_score
    gap = None if denom <= 1e-9 else (retrieved_score - random_score) / denom

    def rate(value: float | None) -> float | None:
        return None if value is None else round(value, 4)

    return {
        "summary": {
            "kind": "test",
            "top_k": config.top_k,
            "retrieved_future_similarity": round(retrieved_score, 4),
            "random_future_similarity": round(random_score, 4),
            "oracle_future_similarity": round(oracle_score, 4),
            "hard_negative_rate": rate(retrieved_hnr),
            "oracle_hard_negative_rate": rate(oracle_hnr),
            "random_hard_negative_rate": rate(random_hnr),
            "gap_closed": None if gap is None else round(float(gap), 4),
            "test_queries": int(retrieved.shape[0]),
        },
        "retrieved": retrieved,
        "oracle": oracle,
        "random": random_index,
        "retrieved_per_query": retrieved_per,
        "oracle_per_query": oracle_per,
        "random_per_query": random_per,
        "retrieved_hnr_per_query": retrieved_hnr_per,
        "oracle_hnr_per_query": oracle_hnr_per,
        "random_hnr_per_query": random_hnr_per,
        "future_similarity": future_similarity,
        "past_similarity": past_similarity,
        "hard_sets": hard_sets,
    }


def rank_retrieval(
    model: RetrieverEncoder,
    database: WindowSet,
    test: WindowSet,
    config: Config,
    device: torch.device,
) -> dict:
    query_embedding = embed(model, test.x, device, batch_size=config.eval_batch_size)
    database_embedding = embed(model, database.x, device, batch_size=config.eval_batch_size)
    similarity = query_embedding @ database_embedding.T
    retrieved = torch.topk(similarity, k=config.top_k, dim=1).indices.numpy()

    future_similarity = window_similarity(test.y, database.y, config.future_temperature)
    past_similarity = window_similarity(test.x, database.x, config.future_temperature)

    oracle = np.argpartition(-future_similarity, config.top_k - 1, axis=1)[:, : config.top_k]
    rng = np.random.default_rng(config.seed)
    random_index = rng.integers(0, len(database), size=(len(test), config.top_k))
    return score_retrieval(retrieved, oracle, random_index, future_similarity, past_similarity, config)


def evaluate_retrieval(
    model: RetrieverEncoder,
    database: WindowSet,
    test: WindowSet,
    config: Config,
    device: torch.device,
) -> dict:
    return rank_retrieval(model, database, test, config, device)["summary"]
