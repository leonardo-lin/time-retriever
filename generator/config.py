"""生成器自己的参数。日期和窗长从检索器 checkpoint 读，不在这里再抄一份。"""

from dataclasses import asdict, dataclass


@dataclass
class Config:
    # 必须指向检索器的 encoder.pt。frozen / none 都要用里面的 config 切数据。
    encoder_path: str = ""
    # frozen：预计算 Top-K。none：K=0，不构造检索器。
    retrieval_source: str = "frozen"
    top_k: int = 5

    d_model: int = 256
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.2
    patch_hours: int = 24

    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    monitor_host: str = "127.0.0.1"
    monitor_port: int = 8766
    run_root: str = "runs/generator"

    def to_dict(self) -> dict:
        return asdict(self)
