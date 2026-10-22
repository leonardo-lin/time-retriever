"""把 21 天负荷压成一个向量，用来和历史案例比相似度。"""

from pathlib import Path

import torch
from torch import nn

from retriever.config import Config

MOMENT_SEQ_LEN = 512


def standardize_windows(windows: torch.Tensor) -> torch.Tensor:
    """按每条窗口自己的均值和标准差缩放，让检索器比较形状而不是绝对负荷。"""
    center = windows.mean(dim=-1, keepdim=True)
    scale = windows.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
    return (windows - center) / scale


def pad_to_moment(windows: torch.Tensor, seq_len: int = MOMENT_SEQ_LEN) -> tuple[torch.Tensor, torch.Tensor]:
    """左边补 0 到 seq_len。mask=0 是补上的位置，不参与 MOMENT 的标准化和池化。"""
    if windows.ndim != 2:
        raise ValueError("windows 必须是 (batch, hours)")
    batch, length = windows.shape
    if length > seq_len:
        raise ValueError(f"窗口 {length} 长于 MOMENT 的 {seq_len}")
    pad = seq_len - length
    x = nn.functional.pad(windows, (pad, 0))
    mask = torch.ones(batch, seq_len, dtype=windows.dtype, device=windows.device)
    if pad:
        mask[:, :pad] = 0
    return x.unsqueeze(1), mask


def _moment_path(config: Config) -> Path:
    path = Path(config.moment_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    if not path.is_dir():
        raise RuntimeError(f"找不到 MOMENT 权重目录：{path}")
    return path


class RetrieverEncoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.encoder_name = config.encoder
        self.freeze_backbone = bool(config.freeze_backbone)
        self.use_lora = bool(config.use_lora)
        if config.encoder == "moment":
            self._init_moment(config)
        elif config.encoder == "scratch":
            self._init_scratch(config)
        else:
            raise ValueError(f"不支持的编码器：{config.encoder}")

    def _init_scratch(self, config: Config) -> None:
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

    def _init_moment(self, config: Config) -> None:
        try:
            from momentfm import MOMENTPipeline
        except ImportError as error:
            raise RuntimeError("缺少 momentfm。在 venv 里执行：pip install momentfm --no-deps && pip install transformers") from error
        backbone = MOMENTPipeline.from_pretrained(
            str(_moment_path(config)),
            model_kwargs={"task_name": "embedding"},
            local_files_only=True,
        )
        backbone.init()
        self.backbone = backbone
        self.output = nn.Linear(int(backbone.config.d_model), config.d_model)
        if self.use_lora:
            if not self.freeze_backbone:
                raise ValueError("use_lora 时请保持 freeze_backbone=True，只训适配器和投影。")
            self._set_backbone_trainable(True)
            self._apply_lora(config)
        else:
            self._set_backbone_trainable(self.freeze_backbone)

    def _set_backbone_trainable(self, freeze: bool) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = not freeze

    def _apply_lora(self, config: Config) -> None:
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as error:
            raise RuntimeError("缺少 peft。在 venv 里执行：pip install peft") from error
        self.backbone.encoder = get_peft_model(
            self.backbone.encoder,
            LoraConfig(
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=["q", "v"],
                bias="none",
            ),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.encoder_name == "moment" and self.freeze_backbone and not self.use_lora:
            self.backbone.eval()
        return self

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        """windows 是原始负荷，形状 (batch, history_hours)。返回单位向量。"""
        if self.encoder_name == "moment":
            return self._forward_moment(windows)
        return self._forward_scratch(windows)

    def _forward_scratch(self, windows: torch.Tensor) -> torch.Tensor:
        windows = standardize_windows(windows)
        batch, length = windows.shape
        patches = windows.reshape(batch, length // self.patch_hours, self.patch_hours)
        hidden = self.patch(patches) + self.position
        hidden = self.encoder(hidden)
        embedding = self.output(hidden.mean(dim=1))
        return nn.functional.normalize(embedding, dim=-1)

    def _forward_moment(self, windows: torch.Tensor) -> torch.Tensor:
        x_enc, mask = pad_to_moment(windows)
        outputs = self.backbone.embed(x_enc=x_enc, input_mask=mask, reduction="mean")
        embedding = self.output(outputs.embeddings)
        return nn.functional.normalize(embedding, dim=-1)
