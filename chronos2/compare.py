"""下载 Chronos-2，对同一份测试查询做零样本预测，按真实负荷从低到高对照。

在项目根目录执行：

    ./venv/bin/python -m chronos2.compare
"""

from __future__ import annotations

import argparse
import json
import webbrowser
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download

from generator.compare import downsample, write_csv
from generator.evaluate import forecast_metrics, sort_by_true
from retriever.config import Config
from retriever.data import load_experiment

MODEL_ID = "amazon/chronos-2"
MODEL_DIR = "models/chronos-2"


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_path(root: Path, path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def model_ready(dest: Path) -> bool:
    return (dest / "config.json").is_file() and (
        (dest / "model.safetensors").is_file() or (dest / "pytorch_model.bin").is_file()
    )


def ensure_model(root: Path, model_path: str) -> Path:
    dest = resolve_path(root, model_path)
    if model_ready(dest):
        print(f"已有本地权重：{dest}")
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    print(f"下载 {MODEL_ID} 到 {dest}…")
    snapshot_download(repo_id=MODEL_ID, local_dir=str(dest))
    if not model_ready(dest):
        raise RuntimeError(f"{dest} 里没有 model.safetensors / pytorch_model.bin，下载不完整。")
    return dest


def stack_point_forecast(means: list) -> np.ndarray:
    """Chronos-2 的 mean 是中位数点预测，每条 (1, horizon)。"""
    return np.stack([np.asarray(item[0], dtype=np.float32) for item in means], axis=0)


def write_html(path: Path, payload: dict) -> None:
    template = (Path(__file__).resolve().parent / "compare.html").read_text(encoding="utf-8")
    path.write_text(template.replace("__DATA__", json.dumps(payload, ensure_ascii=False)), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="用 Chronos-2 零样本预测测试集并对照")
    parser.add_argument("--model-path", default=MODEL_DIR, help="本地权重目录，没有就下载")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config = Config()
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    model_dir = ensure_model(root, args.model_path)
    device = pick_device()
    print(f"加载 Chronos-2（{device}）…")
    from chronos import Chronos2Pipeline

    pipeline = Chronos2Pipeline.from_pretrained(str(model_dir), device_map=device.type)

    print("读取负荷并切分窗口…")
    _database, _train, _val, test, summary = load_experiment(config, root)
    print(f"测试 {summary['test_cases']} 条：{summary['test_start']} → {summary['test_end']}")

    context = np.ascontiguousarray(test.x[:, None, :], dtype=np.float32)
    print(f"在 {len(test)} 条测试查询上零样本预测…")
    _quantiles, means = pipeline.predict_quantiles(
        context,
        prediction_length=config.horizon_hours,
        quantile_levels=[0.5],
        batch_size=args.batch_size,
    )
    y_hat = stack_point_forecast(means)
    metrics = forecast_metrics(test.y, y_hat, config.patch_hours)
    ranked = sort_by_true(test.y, y_hat, test.origin_utc)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = root / "runs" / "chronos2" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "test_sorted.csv"
    html_path = out_dir / "test_sorted.html"
    write_csv(csv_path, ranked)
    write_html(
        html_path,
        {
            "queries": len(test),
            "points": int(ranked["true"].size),
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "mape": metrics["mape"],
            "day_mae": metrics["day_mae"],
            "day_rmse": metrics["day_rmse"],
            "model": "chronos-2",
            "true": downsample(ranked["true"]),
            "pred": downsample(ranked["pred"]),
        },
    )
    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "model_id": MODEL_ID,
                "model_path": str(model_dir),
                "device": str(device),
                "batch_size": args.batch_size,
                "test_cases": len(test),
                **metrics,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    days = "  ".join(
        f"第{i + 1}天 MAE {mae:.4f} RMSE {rmse:.4f}"
        for i, (mae, rmse) in enumerate(zip(metrics["day_mae"], metrics["day_rmse"]))
    )
    print(
        f"测试 MAE {metrics['mae']:.4f}  RMSE {metrics['rmse']:.4f}  "
        f"MAPE {metrics['mape']:.4f}%"
    )
    print(days)
    print(f"已按真实值从低到高排序：{csv_path}")
    print(f"对照页：{html_path}")
    if not args.no_browser:
        webbrowser.open(html_path.resolve().as_uri())


if __name__ == "__main__":
    main()
