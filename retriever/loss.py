"""第 16 节的两个损失：对比学习，以及按未来有用程度拉开的排序。"""

import torch
import torch.nn.functional as F


def cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """两边都已经是单位向量，点积就是余弦相似度。"""
    return (left * right).sum(dim=-1)


def retriever_losses(
    query: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    temperature: float,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """query (B, D)，positive (B, D)，negative (B, K, D)。"""
    positive_score = cosine(query, positive)
    negative_score = cosine(query.unsqueeze(1), negative)
    logits = torch.cat([positive_score.unsqueeze(1), negative_score], dim=1) / temperature
    target = torch.zeros(query.shape[0], dtype=torch.long, device=query.device)
    contrastive = F.cross_entropy(logits, target)
    ranking = F.relu(margin - positive_score.unsqueeze(1) + negative_score).mean()
    return contrastive, ranking
