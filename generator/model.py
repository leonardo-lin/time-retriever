"""用当前 X 和检索到的历史案例生成 7 天点预测。"""

import torch
from torch import nn

from generator.config import Config


class ForecastGenerator(nn.Module):
    def __init__(self, config: Config, history_hours: int, horizon_hours: int):
        super().__init__()
        if history_hours % config.patch_hours != 0:
            raise ValueError("history_hours 必须能被 patch_hours 整除")
        if horizon_hours % config.patch_hours != 0:
            raise ValueError("horizon_hours 必须能被 patch_hours 整除")
        if config.d_model % config.num_heads != 0:
            raise ValueError("d_model 必须能被 num_heads 整除")
        self.patch_hours = config.patch_hours
        self.history_hours = history_hours
        self.horizon_hours = horizon_hours
        n_history = history_hours // config.patch_hours
        n_horizon = horizon_hours // config.patch_hours
        self.n_history = n_history
        self.n_horizon = n_horizon
        self.patch = nn.Linear(config.patch_hours, config.d_model)
        self.xi_pool = nn.Linear(config.d_model, config.d_model)
        self.day_query = nn.Parameter(torch.randn(1, n_horizon, config.d_model) * 0.02)
        self.x_pos = nn.Parameter(torch.randn(1, n_history, config.d_model) * 0.02)
        self.y_pos = nn.Parameter(torch.randn(1, n_horizon, config.d_model) * 0.02)
        self.xi_type = nn.Parameter(torch.randn(1, 1, config.d_model) * 0.02)
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
        self.head = nn.Linear(config.d_model, config.patch_hours)

    def forward(
        self,
        x: torch.Tensor,
        retrieved_x: torch.Tensor,
        retrieved_y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x (B, history)，retrieved_x (B, K, history)，retrieved_y (B, K, horizon)。没有当前 Y。"""
        center = x.mean(dim=-1, keepdim=True)
        scale = x.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
        x_n = (x - center) / scale
        x_tokens = self.patch(x_n.reshape(x.shape[0], self.n_history, self.patch_hours)) + self.x_pos
        pieces = [self.day_query.expand(x.shape[0], -1, -1), x_tokens]
        k = retrieved_x.shape[1]
        if k:
            stats = center.unsqueeze(1)
            spread = scale.unsqueeze(1)
            xi_n = (retrieved_x - stats) / spread
            yi_n = (retrieved_y - stats) / spread
            xi_tokens = self.xi_pool(
                self.patch(xi_n.reshape(x.shape[0], k, self.n_history, self.patch_hours)).mean(dim=2)
            ) + self.xi_type
            yi_tokens = self.patch(yi_n.reshape(x.shape[0], k, self.n_horizon, self.patch_hours)) + self.y_pos
            cases = torch.cat([xi_tokens.unsqueeze(2), yi_tokens], dim=2)
            pieces.append(cases.reshape(x.shape[0], k * (1 + self.n_horizon), -1))
        hidden = self.encoder(torch.cat(pieces, dim=1))
        y_hat = self.head(hidden[:, : self.n_horizon]).reshape(x.shape[0], self.horizon_hours)
        return y_hat, center, scale
