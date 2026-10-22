"""把 21 天负荷压成一个向量，用来和历史案例比相似度。"""

import torch
from torch import nn

from retriever.config import Config


def standardize_windows(windows: torch.Tensor) -> torch.Tensor:
    """按每条窗口自己的均值和标准差缩放，让检索器比较形状而不是绝对负荷。"""
    center = windows.mean(dim=-1, keepdim=True)
    scale = windows.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
    return (windows - center) / scale


class RetrieverEncoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        if config.history_hours % config.patch_hours != 0:
            raise ValueError("history_hours 必须能被 patch_hours 整除")
        if config.d_model % config.num_heads != 0:
            raise ValueError("d_model 必须能被 num_heads 整除")
        n_patches = config.history_hours // config.patch_hours
        self.patch_hours = config.patch_hours
        self.patch = nn.Linear(config.patch_hours, config.d_model)
        self.position = nn.Parameter(torch.randn(1, n_patches, config.d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.num_layers, enable_nested_tensor=False)
        self.output = nn.Linear(config.d_model, config.d_model)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        """windows 是原始负荷，形状 (batch, history_hours)。内部按窗口标准化后返回单位向量。"""
        windows = standardize_windows(windows)
        batch, length = windows.shape
        patches = windows.reshape(batch, length // self.patch_hours, self.patch_hours)
        hidden = self.patch(patches) + self.position
        hidden = self.encoder(hidden)
        embedding = self.output(hidden.mean(dim=1))
        return nn.functional.normalize(embedding, dim=-1)
