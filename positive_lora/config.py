"""这一项实验自己的参数。负例相关字段不出现。"""

from dataclasses import asdict, dataclass

from retriever.config import Config as RetrieverConfig


@dataclass
class Config:
    csv_path: str = "data/caiso_demand_forecast/SLD_FCST_ACTUAL.csv"
    tac_area: str = "CA ISO-TAC"

    history_hours: int = 21 * 24
    horizon_hours: int = 7 * 24

    database_before: str = "2025-01-01"
    train_from: str = "2025-01-01"
    eval_from: str = "2026-01-01"
    stride_hours: int = 24
    val_ratio: float = 0.5
    test_ratio: float = 0.5

    positive_future_quantile: float = 0.95
    positive_min_similarity: float = 0.75
    future_temperature: float = 1.0

    moment_path: str = "models/MOMENT-1-large"
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    adapter_learning_rate: float = 1e-5
    d_model: int = 128
    patch_hours: int = 24

    batch_size: int = 8
    eval_batch_size: int = 8
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    seed: int = 42
    top_k: int = 5

    monitor_host: str = "127.0.0.1"
    monitor_port: int = 8767
    run_root: str = "runs/positive_lora"

    def to_dict(self) -> dict:
        return asdict(self)


def to_retriever_config(config: Config) -> RetrieverConfig:
    """给 load_experiment / RetrieverEncoder / evaluate 用。LoRA 和冻主干在这里锁死。"""
    return RetrieverConfig(
        csv_path=config.csv_path,
        tac_area=config.tac_area,
        history_hours=config.history_hours,
        horizon_hours=config.horizon_hours,
        database_before=config.database_before,
        train_from=config.train_from,
        eval_from=config.eval_from,
        stride_hours=config.stride_hours,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        positive_future_quantile=config.positive_future_quantile,
        positive_min_similarity=config.positive_min_similarity,
        future_temperature=config.future_temperature,
        encoder="moment",
        moment_path=config.moment_path,
        freeze_backbone=True,
        use_lora=True,
        lora_rank=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        d_model=config.d_model,
        patch_hours=config.patch_hours,
        batch_size=config.batch_size,
        eval_batch_size=config.eval_batch_size,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        backbone_learning_rate=config.adapter_learning_rate,
        weight_decay=config.weight_decay,
        grad_clip=config.grad_clip,
        seed=config.seed,
        top_k=config.top_k,
    )
