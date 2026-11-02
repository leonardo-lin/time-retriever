"""用当前 X 和检索到的历史案例生成 7 天点预测。"""

from pathlib import Path

import torch
from torch import nn

from generator.config import Config
from retriever.model import pad_to_moment


def _moment_path(path: str) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = Path(__file__).resolve().parents[1] / resolved
    if not resolved.is_dir():
        raise RuntimeError(f"找不到 MOMENT 权重目录：{resolved}")
    return resolved


class ForecastGenerator(nn.Module):
    def __init__(self, config: Config, history_hours: int, horizon_hours: int):
        super().__init__()
        self.backbone_name = config.backbone
        self.freeze_backbone = bool(config.freeze_backbone)
        self.history_hours = history_hours
        self.horizon_hours = horizon_hours
        self.patch_hours = config.patch_hours
        if config.backbone == "moment":
            self._init_moment(config, horizon_hours)
        elif config.backbone == "scratch":
            self._init_scratch(config, history_hours, horizon_hours)
        else:
            raise ValueError(f"不支持的生成器骨干：{config.backbone}")

    def _init_scratch(self, config: Config, history_hours: int, horizon_hours: int) -> None:
        if history_hours % config.patch_hours != 0:
            raise ValueError("history_hours 必须能被 patch_hours 整除")
        if horizon_hours % config.patch_hours != 0:
            raise ValueError("horizon_hours 必须能被 patch_hours 整除")
        if config.d_model % config.num_heads != 0:
            raise ValueError("d_model 必须能被 num_heads 整除")
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

    def _init_moment(self, config: Config, horizon_hours: int) -> None:
        try:
            from momentfm import MOMENTPipeline
        except ImportError as error:
            raise RuntimeError("缺少 momentfm。在 venv 里执行：pip install momentfm --no-deps && pip install transformers") from error
        backbone = MOMENTPipeline.from_pretrained(
            str(_moment_path(config.moment_path)),
            model_kwargs={"task_name": "forecasting", "forecast_horizon": horizon_hours},
            local_files_only=True,
        )
        backbone.init()
        self.backbone = backbone
        # Xi 的均值决定每条 Yi 的权重，再映成残差。预测头是随机初始化的，必须训。
        self.case_score = nn.Linear(1, 1)
        self.case_mix = nn.Linear(horizon_hours, horizon_hours)
        if self.freeze_backbone:
            for name, parameter in self.backbone.named_parameters():
                parameter.requires_grad = name.startswith("head.")

    def train(self, mode: bool = True):
        super().train(mode)
        if self.backbone_name == "moment" and self.freeze_backbone:
            self.backbone.normalizer.eval()
            self.backbone.tokenizer.eval()
            self.backbone.patch_embedding.eval()
            self.backbone.encoder.eval()
        return self

    def forward(
        self,
        x: torch.Tensor,
        retrieved_x: torch.Tensor,
        retrieved_y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x (B, history)，retrieved_x (B, K, history)，retrieved_y (B, K, horizon)。没有当前 Y。"""
        center = x.mean(dim=-1, keepdim=True)
        scale = x.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
        if self.backbone_name == "moment":
            return self._forward_moment(x, retrieved_x, retrieved_y, center, scale)
        return self._forward_scratch(x, retrieved_x, retrieved_y, center, scale)

    def _forward_scratch(
        self,
        x: torch.Tensor,
        retrieved_x: torch.Tensor,
        retrieved_y: torch.Tensor,
        center: torch.Tensor,
        scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

    def _forward_moment(
        self,
        x: torch.Tensor,
        retrieved_x: torch.Tensor,
        retrieved_y: torch.Tensor,
        center: torch.Tensor,
        scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_enc, mask = pad_to_moment(x)
        forecast = self.backbone(x_enc=x_enc, input_mask=mask).forecast.squeeze(1)
        y_hat = (forecast - center) / scale
        k = retrieved_x.shape[1]
        if k:
            stats = center.unsqueeze(1)
            spread = scale.unsqueeze(1)
            xi_n = (retrieved_x - stats) / spread
            yi_n = (retrieved_y - stats) / spread
            weights = torch.softmax(self.case_score(xi_n.mean(dim=-1, keepdim=True)).squeeze(-1), dim=1)
            pooled = (weights.unsqueeze(-1) * yi_n).sum(dim=1)
            y_hat = y_hat + self.case_mix(pooled)
        return y_hat, center, scale
