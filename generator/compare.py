"""用训好的生成器跑完全部测试查询，按真实负荷从低到高排序后对照预测。

在项目根目录执行：

    ./venv/bin/python -m generator.compare --generator runs/generator/<run>/generator.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import webbrowser
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

from generator.config import Config
from generator.evaluate import forecast_metrics, predict_forecast, sort_by_true
from generator.model import ForecastGenerator
from generator.retrieve import (
    empty_retrieved,
    load_frozen_encoder,
    read_checkpoint,
    retrieve_oracle,
    retrieve_queries,
    retriever_config_from_checkpoint,
)
from generator.train import pick_device, resolve_path
from retriever.data import format_pacific, load_experiment


def generator_config_from_checkpoint(checkpoint: dict) -> Config:
    if "config" not in checkpoint:
        raise RuntimeError("generator.pt 里没有 config。")
    allowed = {item.name for item in fields(Config)}
    payload = {key: value for key, value in checkpoint["config"].items() if key in allowed}
    return Config(**payload)


def resolve_top_k(saved_config: dict, source: str, override: int | None) -> int:
    """用这次生成器训练时的 K。不要回落到现在 config.py 的默认值。"""
    if source == "none":
        return 0
    if override is not None:
        return int(override)
    if "top_k" not in saved_config:
        raise RuntimeError("generator.pt 里没有 top_k，请用 --top-k 指定这次训练用的 K。")
    return int(saved_config["top_k"])


def downsample(values: np.ndarray, cap: int = 4000) -> list[float]:
    # shortcut: 网页只画均匀抽样，完整对照看 CSV；点太多时再改成按分位聚合
    if len(values) <= cap:
        picked = values
    else:
        index = np.linspace(0, len(values) - 1, cap).astype(np.int64)
        picked = values[index]
    return [round(float(item), 4) for item in picked]


def write_csv(path: Path, ranked: dict[str, np.ndarray]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "true_mw", "pred_mw", "error_mw", "query", "hour", "origin"])
        origins = [format_pacific(stamp) for stamp in ranked["origin_utc"]]
        for rank, true, pred, query, hour, origin in zip(
            range(len(ranked["true"])),
            ranked["true"],
            ranked["pred"],
            ranked["query"],
            ranked["hour"],
            origins,
        ):
            writer.writerow([
                rank,
                f"{float(true):.4f}",
                f"{float(pred):.4f}",
                f"{float(pred - true):.4f}",
                int(query),
                int(hour),
                origin,
            ])


def write_html(path: Path, payload: dict) -> None:
    template = (Path(__file__).resolve().parent / "compare.html").read_text(encoding="utf-8")
    path.write_text(template.replace("__DATA__", json.dumps(payload, ensure_ascii=False)), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按真实负荷从低到高对照生成器测试预测")
    parser.add_argument("--generator", required=True, help="生成器 generator.pt 路径")
    parser.add_argument("--top-k", type=int, help="覆盖 checkpoint 里的 K；默认用训练时的值")
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    generator_path = resolve_path(root, args.generator)
    if not generator_path.is_file():
        raise RuntimeError(f"找不到生成器权重：{generator_path}")

    device = pick_device()
    stored = torch.load(generator_path, map_location=device, weights_only=False)
    if not isinstance(stored, dict) or "model" not in stored:
        raise RuntimeError(f"{generator_path} 里没有 model 权重。")
    if "retriever_config" not in stored:
        raise RuntimeError("generator.pt 里没有 retriever_config，无法复原数据切分。")
    config = generator_config_from_checkpoint(stored)
    k = resolve_top_k(stored["config"], config.retrieval_source, args.top_k)
    config.top_k = k
    retriever_config = retriever_config_from_checkpoint({"config": stored["retriever_config"]})
    np.random.seed(retriever_config.seed)
    torch.manual_seed(retriever_config.seed)

    print("读取负荷并切分窗口…")
    database, _train, _val, test, summary = load_experiment(retriever_config, root)
    print(f"测试 {summary['test_cases']} 条：{summary['test_start']} → {summary['test_end']}")
    print(f"这份生成器 retrieval_source={config.retrieval_source}，K={k}")

    if config.retrieval_source == "frozen":
        if not config.encoder_path:
            raise RuntimeError("这个生成器是冻结检索训的，但 checkpoint 里没有 encoder_path。")
        encoder_path = resolve_path(root, config.encoder_path)
        print("加载冻结检索器并预计算 Top-K…")
        encoder = load_frozen_encoder(read_checkpoint(encoder_path), retriever_config, device)
        test_hit = retrieve_queries(
            encoder,
            database,
            test,
            k,
            device,
            retriever_config.eval_batch_size,
        )
        del encoder
        if device.type == "cuda":
            torch.cuda.empty_cache()
    elif config.retrieval_source == "oracle":
        print("按未来相似度取正解 Top-K…")
        test_hit = retrieve_oracle(test, database, k, retriever_config.future_temperature)
    elif config.retrieval_source == "none":
        test_hit = empty_retrieved(len(test))
        print("无检索基线：K=0。")
    else:
        raise RuntimeError(f"不支持的 retrieval_source：{config.retrieval_source}")

    model = ForecastGenerator(config, retriever_config.history_hours, retriever_config.horizon_hours).to(device)
    model.load_state_dict(stored["model"])
    print(f"在 {len(test)} 条测试查询上预测…")
    y_hat = predict_forecast(model, test, database, test_hit, config, device)
    metrics = forecast_metrics(test.y, y_hat, config.patch_hours)
    ranked = sort_by_true(test.y, y_hat, test.origin_utc)

    out_dir = generator_path.parent
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
            "retrieval_source": config.retrieval_source,
            "top_k": k,
            "true": downsample(ranked["true"]),
            "pred": downsample(ranked["pred"]),
        },
    )
    days = "  ".join(
        f"第{i + 1}天 MAE {mae:.4f} RMSE {rmse:.4f}"
        for i, (mae, rmse) in enumerate(zip(metrics["day_mae"], metrics["day_rmse"]))
    )
    print(
        f"测试 MAE {metrics['mae']:.4f}  RMSE {metrics['rmse']:.4f}  "
        f"MAPE {metrics['mape']:.4f}%  K={k}"
    )
    print(days)
    print(f"已按真实值从低到高排序：{csv_path}")
    print(f"对照页：{html_path}")
    if not args.no_browser:
        webbrowser.open(html_path.resolve().as_uri())


if __name__ == "__main__":
    main()
