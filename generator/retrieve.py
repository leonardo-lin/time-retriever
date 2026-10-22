"""加载冻结检索器，预计算 Top-K 行号。不读当前查询的 Y。"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import torch

from retriever.config import Config as RetrieverConfig
from retriever.data import WindowSet
from retriever.evaluate import embed
from retriever.model import RetrieverEncoder


@dataclass
class Retrieved:
    """每条查询对应 K 个检索库行号。K=0 时第二维是 0。"""

    indices: np.ndarray


def empty_retrieved(n_queries: int) -> Retrieved:
    return Retrieved(np.zeros((n_queries, 0), dtype=np.int64))


def read_checkpoint(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"找不到检索器权重：{path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "config" not in checkpoint:
        raise RuntimeError(f"{path} 里没有 config，无法复原数据切分。")
    return checkpoint


def retriever_config_from_checkpoint(checkpoint: dict) -> RetrieverConfig:
    allowed = {item.name for item in fields(RetrieverConfig)}
    payload = {key: value for key, value in checkpoint["config"].items() if key in allowed}
    return RetrieverConfig(**payload)


def load_frozen_encoder(checkpoint: dict, config: RetrieverConfig, device: torch.device) -> RetrieverEncoder:
    model = RetrieverEncoder(config)
    if "model" not in checkpoint:
        raise RuntimeError("encoder.pt 里没有 model 权重。")
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def retrieve_splits(
    model: RetrieverEncoder,
    database: WindowSet,
    train: WindowSet,
    val: WindowSet,
    test: WindowSet,
    k: int,
    device: torch.device,
    batch_size: int,
) -> tuple[Retrieved, Retrieved, Retrieved]:
    if k <= 0:
        return empty_retrieved(len(train)), empty_retrieved(len(val)), empty_retrieved(len(test))
    if k > len(database):
        raise RuntimeError(f"K={k} 大于检索库 {len(database)} 条。")
    database_z = embed(model, database.x, device, batch_size=batch_size)

    def one(queries: WindowSet) -> Retrieved:
        query_z = embed(model, queries.x, device, batch_size=batch_size)
        indices = torch.topk(query_z @ database_z.T, k=k, dim=1).indices.cpu().numpy().astype(np.int64)
        return Retrieved(indices)

    return one(train), one(val), one(test)


def gather_cases(database: WindowSet, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """indices (B, K) -> Xi (B, K, history), Yi (B, K, horizon)。"""
    return database.x[indices], database.y[indices]
