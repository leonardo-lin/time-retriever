"""训练负荷预测生成器，并在本地网页上实时画误差。

在项目根目录执行：

    ./venv/bin/python -m generator.train --encoder runs/retriever/<run>/encoder.pt
"""

from __future__ import annotations

import argparse
import json
import threading
import webbrowser
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from generator.config import Config
from generator.evaluate import evaluate_forecast
from generator.model import ForecastGenerator
from generator.monitor import start_monitor
from generator.retrieve import (
    Retrieved,
    empty_retrieved,
    gather_cases,
    load_frozen_encoder,
    read_checkpoint,
    retrieve_splits,
    retriever_config_from_checkpoint,
)
from retriever.config import Config as RetrieverConfig
from retriever.data import WindowSet, load_experiment


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_path(root: Path, path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def append_metric(path: Path, record: dict) -> None:
    record["time"] = datetime.now().isoformat(timespec="seconds")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_model(path: Path, model: ForecastGenerator, config: Config, retriever_config: RetrieverConfig) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "config": config.to_dict(),
            "retriever_config": retriever_config.to_dict(),
        },
        path,
    )


def batch_mse(
    model: ForecastGenerator,
    queries: WindowSet,
    database: WindowSet,
    retrieved: Retrieved,
    indices: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    x = torch.from_numpy(np.ascontiguousarray(queries.x[indices])).to(device)
    y = torch.from_numpy(np.ascontiguousarray(queries.y[indices])).to(device)
    xi, yi = gather_cases(database, retrieved.indices[indices])
    xi_t = torch.from_numpy(np.ascontiguousarray(xi)).to(device)
    yi_t = torch.from_numpy(np.ascontiguousarray(yi)).to(device)
    y_hat, center, scale = model(x, xi_t, yi_t)
    return F.mse_loss(y_hat, (y - center) / scale)


def run_epoch(
    model: ForecastGenerator,
    queries: WindowSet,
    database: WindowSet,
    retrieved: Retrieved,
    config: Config,
    device: torch.device,
    rng: np.random.Generator,
    optimizer: torch.optim.Optimizer | None,
) -> list[float]:
    training = optimizer is not None
    model.train(training)
    order = rng.permutation(len(queries))
    recorded = []
    for start in range(0, len(order), config.batch_size):
        indices = order[start : start + config.batch_size]
        loss = batch_mse(model, queries, database, retrieved, indices, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
        recorded.append(float(loss.detach()))
    return recorded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练检索增强负荷预测生成器")
    parser.add_argument("--encoder", type=str, help="检索器 encoder.pt 路径")
    parser.add_argument("--source", choices=("frozen", "none"))
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--port", type=int)
    parser.add_argument("--exit-when-done", action="store_true")
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config()
    if args.encoder is not None:
        config.encoder_path = args.encoder
    if args.source is not None:
        config.retrieval_source = args.source
    if args.top_k is not None:
        config.top_k = args.top_k
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.port is not None:
        config.monitor_port = args.port
    if config.retrieval_source not in ("frozen", "none"):
        raise RuntimeError(f"不支持的 retrieval_source：{config.retrieval_source}")
    if not config.encoder_path:
        raise RuntimeError("请用 --encoder 指向检索器的 encoder.pt。")

    root = Path(__file__).resolve().parents[1]
    encoder_path = resolve_path(root, config.encoder_path)
    checkpoint = read_checkpoint(encoder_path)
    retriever_config = retriever_config_from_checkpoint(checkpoint)
    np.random.seed(retriever_config.seed)
    torch.manual_seed(retriever_config.seed)
    device = pick_device()

    print("读取负荷并切分窗口…")
    database, train, val, test, summary = load_experiment(retriever_config, root)
    print(
        f"检索库 {summary['database_cases']} 条，"
        f"未来结束于 {summary['database_future_end']}"
    )
    print(f"训练 {summary['train_cases']} 条：{summary['train_start']} → {summary['train_end']}")
    print(f"验证 {summary['val_cases']} 条：{summary['val_start']} → {summary['val_end']}")
    print(f"测试 {summary['test_cases']} 条：{summary['test_start']} → {summary['test_end']}")
    print(f"检索库未来与第一条查询相隔 {summary['leakage_gap_hours']} 小时")

    if config.retrieval_source == "frozen":
        print("加载冻结检索器并预计算 Top-K…")
        encoder = load_frozen_encoder(checkpoint, retriever_config, device)
        train_hit, val_hit, test_hit = retrieve_splits(
            encoder,
            database,
            train,
            val,
            test,
            config.top_k,
            device,
            retriever_config.eval_batch_size,
        )
        del encoder
        if device.type == "cuda":
            torch.cuda.empty_cache()
        k = config.top_k
    else:
        train_hit, val_hit, test_hit = (
            empty_retrieved(len(train)),
            empty_retrieved(len(val)),
            empty_retrieved(len(test)),
        )
        k = 0
        print("无检索基线：不加载编码器，K=0。")

    run_dir = root / config.run_root / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    summary["device"] = str(device)
    summary["run_dir"] = str(run_dir)
    summary["retrieval_source"] = config.retrieval_source
    summary["top_k"] = k
    summary["encoder_path"] = str(encoder_path)
    (run_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    metrics_path = run_dir / "metrics.jsonl"

    server, port = start_monitor(run_dir, config.monitor_host, config.monitor_port)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://{config.monitor_host}:{port}"
    print(f"监控页面：{url}")
    if not args.no_browser:
        webbrowser.open(url)

    model = ForecastGenerator(config, retriever_config.history_hours, retriever_config.horizon_hours).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    train_rng = np.random.default_rng(retriever_config.seed)
    best_val = float("inf")
    best_epoch = 0
    step = 0
    try:
        for epoch in range(1, config.epochs + 1):
            records = run_epoch(model, train, database, train_hit, config, device, train_rng, optimizer)
            for loss in records:
                step += 1
                append_metric(metrics_path, {
                    "kind": "step",
                    "epoch": epoch,
                    "step": step,
                    "loss": round(loss, 6),
                })
            val_rng = np.random.default_rng(retriever_config.seed)
            with torch.no_grad():
                val_records = run_epoch(model, val, database, val_hit, config, device, val_rng, None)
            train_loss = float(np.mean(records)) if records else float("nan")
            val_loss = float(np.mean(val_records)) if val_records else float("nan")
            val_forecast = evaluate_forecast(model, val, database, val_hit, config, device)
            val_mae = val_forecast["mae"]
            append_metric(metrics_path, {
                "kind": "epoch",
                "epoch": epoch,
                "step": step,
                "train_loss": None if np.isnan(train_loss) else round(train_loss, 6),
                "val_loss": None if np.isnan(val_loss) else round(val_loss, 6),
                "val_mae": val_mae,
            })
            print(
                f"第 {epoch:02d} 轮  训练 MSE {train_loss:.4f}  验证 MSE {val_loss:.4f}  "
                f"验证 MAE {val_mae:.4f}"
            )
            save_model(run_dir / "generator_last.pt", model, config, retriever_config)
            if val_mae < best_val:
                best_val = val_mae
                best_epoch = epoch
                save_model(run_dir / "generator.pt", model, config, retriever_config)

        print("在测试查询上评估预测…")
        stored = torch.load(run_dir / "generator.pt", map_location=device, weights_only=False)
        model.load_state_dict(stored["model"])
        result = evaluate_forecast(model, test, database, test_hit, config, device)
        append_metric(metrics_path, result)
        days = "  ".join(f"第{i + 1}天 {mae:.4f}" for i, mae in enumerate(result["day_mae"]))
        print(
            f"测试 MAE {result['mae']:.4f}  RMSE {result['rmse']:.4f}  "
            f"MAPE {result['mape']:.4f}%  K={result['top_k']}"
        )
        print(days)
        print(f"最好的模型：第 {best_epoch} 轮，验证 MAE {best_val:.4f}，{run_dir / 'generator.pt'}")
    except KeyboardInterrupt:
        save_model(run_dir / "generator_last.pt", model, config, retriever_config)
        print("\n训练已中断，当前模型写在 generator_last.pt。")
    finally:
        if args.exit_when_done:
            server.shutdown()

    if args.exit_when_done:
        return
    print("监控页面保持打开。再按 Ctrl+C 结束。")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        print("\n已关闭。")
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
