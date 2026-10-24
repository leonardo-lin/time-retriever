"""全部可调参数。改实验时优先改这个文件。"""

from dataclasses import asdict, dataclass


@dataclass
class Config:
    # CAISO 全系统负荷。CSV 里还有 PGE、SCE 等分区，这次不用。
    csv_path: str = "data/caiso_demand_forecast/SLD_FCST_ACTUAL.csv"
    tac_area: str = "CA ISO-TAC"

    # 用过去 21 天预测未来 7 天。检索器只读这 21 天；7 天只用来制造监督。
    history_hours: int = 21 * 24
    horizon_hours: int = 7 * 24

    # 按加州本地时间的零点切。检索库未来必须在 2025 年以前结束，避免碰到训练查询。
    # 训练查询：预测起点落在 2025 年。验证/测试：从 2026 年查询里随机抽。
    database_before: str = "2025-01-01"
    train_from: str = "2025-01-01"
    eval_from: str = "2026-01-01"

    # 24 表示每个加州本地日的 23:00 取一条。改成 1 就是每小时一条。
    stride_hours: int = 24

    # 只用来切 2026 年查询。0.5 / 0.5 表示全部用上，打乱后对半做验证和测试。
    val_ratio: float = 0.5
    test_ratio: float = 0.5

    # 正例：这条查询自己的未来相似度前 5%，且绝对值够高。
    # 0.75 约等于冬季查询 95 分位的均值；只靠分位会把「相对不差」的案例当正例。
    positive_future_quantile: float = 0.95
    positive_min_similarity: float = 0.75
    low_future_quantile: float = 0.50
    high_past_quantile: float = 0.80
    low_past_quantile: float = 0.50

    # 把未来距离变成相关度：S = exp(-distance / temperature)。
    future_temperature: float = 1.0
    # 对比损失里的温度，作用在余弦相似度上。
    contrastive_temperature: float = 0.1
    margin: float = 0.2
    lambda_contrastive: float = 1.0
    lambda_ranking: float = 1.0

    # moment：本地 MOMENT-1-large + 一层投影。scratch：从零训练的小 Transformer。
    encoder: str = "moment"
    moment_path: str = "models/MOMENT-1-large"
    # 主干权重冻住。use_lora 时只在 T5 的 q/v 上加适配器。
    freeze_backbone: bool = True
    use_lora: bool = True
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_learning_rate: float = 1e-5
    # 默认从原始 MOMENT 起训。只有要接某次投影时才填 encoder.pt。
    init_from: str = ""
    d_model: int = 128
    num_layers: int = 3
    num_heads: int = 8
    dropout: float = 0.2
    # scratch 用：每 24 小时合成一个 token。21 天就是 21 个 token。必须能整除 history_hours。
    patch_hours: int = 24

    batch_size: int = 8
    eval_batch_size: int = 8
    hard_negatives: int = 2
    easy_negatives: int = 2
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    seed: int = 42
    top_k: int = 5

    monitor_host: str = "127.0.0.1"
    monitor_port: int = 8765
    run_root: str = "runs/retriever"

    def to_dict(self) -> dict:
        return asdict(self)
