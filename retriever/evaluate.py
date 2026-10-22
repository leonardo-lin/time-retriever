"""训练结束后，检查检索器找到的案例未来是否更相似。"""

from __future__ import annotations

import numpy as np
import torch

from retriever.config import Config
from retriever.data import WindowSet
from retriever.model import RetrieverEncoder
from retriever.sampling import classify_candidates, window_similarity


def embed(model: RetrieverEncoder, windows: np.ndarray, device: torch.device, batch_size: int = 256) -> torch.Tensor:
    model.eval()
    pieces: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            batch = torch.from_numpy(np.ascontiguousarray(windows[start : start + batch_size])).to(device)
            pieces.append(model(batch).cpu())
    return torch.cat(pieces, dim=0)


def evaluate_retrieval(
    model: RetrieverEncoder,
    database: WindowSet,
    test: WindowSet,
    config: Config,
    device: torch.device,
) -> dict:
    query_embedding = embed(model, test.x, device)
    database_embedding = embed(model, database.x, device)
    similarity = query_embedding @ database_embedding.T
    retrieved = torch.topk(similarity, k=config.top_k, dim=1).indices.numpy()

    future_similarity = window_similarity(test.y, database.y, config.future_temperature)
    past_similarity = window_similarity(test.x, database.x, config.future_temperature)
    rows = np.arange(len(test))[:, None]
    retrieved_score = float(future_similarity[rows, retrieved].mean())

    oracle = np.argpartition(-future_similarity, config.top_k - 1, axis=1)[:, : config.top_k]
    oracle_score = float(future_similarity[rows, oracle].mean())

    rng = np.random.default_rng(config.seed)
    random_index = rng.integers(0, len(database), size=(len(test), config.top_k))
    random_score = float(future_similarity[rows, random_index].mean())

    hard_hits = []
    for row in range(len(test)):
        _, hard, _ = classify_candidates(past_similarity[row], future_similarity[row], config)
        if len(hard) == 0:
            continue
        hard_set = set(hard.tolist())
        hard_hits.append(sum(int(index in hard_set) for index in retrieved[row]) / config.top_k)

    return {
        "kind": "test",
        "top_k": config.top_k,
        "retrieved_future_similarity": round(retrieved_score, 4),
        "random_future_similarity": round(random_score, 4),
        "oracle_future_similarity": round(oracle_score, 4),
        "hard_negative_rate": None if not hard_hits else round(float(np.mean(hard_hits)), 4),
        "test_queries": len(test),
    }
