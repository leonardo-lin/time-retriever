"""用正例把查询向量拉近历史向量，只训 LoRA 和投影。

在项目根目录执行：

    ./venv/bin/python -m positive_lora.train
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

from positive_lora.config import Config, to_retriever_config
from positive_lora.monitor import start_monitor
from positive_lora.positives import build_positives, positive_report
from retriever.data import WindowSet, load_experiment
from retriever.evaluate import evaluate_retrieval
from retriever.model import RetrieverEncoder


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def encode(model: RetrieverEncoder, windows: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(windows)).to(device)
    return model(tensor)


def positive_loss(
    model: RetrieverEncoder,
    queries: WindowSet,
    database: WindowSet,
    pools: list[np.ndarray],
    indices: np.ndarray,
    rng: np.random.Generator,
    device: torch.device,
) -> torch.Tensor:
    positives = [int(rng.choice(pools[int(index)])) for index in indices]
    packed = np.concatenate([queries.x[indices], database.x[np.array(positives)]])
    encoded = encode(model, packed, device)
    query_z = encoded[: len(indices)]
    positive_z = encoded[len(indices) :]
    return 1.0 - (query_z * positive_z).sum(dim=-1).mean()


def run_epoch(
    model: RetrieverEncoder,
    queries: WindowSet,
    database: WindowSet,
    pools: list[np.ndarray],
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
        loss = positive_loss(model, queries, database, pools, indices, rng, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
        recorded.append(float(loss.detach()))
    return recorded


def append_metric(path: Path, record: dict) -> None:
    record["time"] = datetime.now().isoformat(timespec="seconds")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_model(path: Path, model: RetrieverEncoder, config: Config) -> None:
    torch.save({"model": model.state_dict(), "config": config.to_dict()}, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="用正例 LoRA 微调负荷检索器")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--port", type=int)
    parser.add_argument("--exit-when-done", action="store_true")
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config()
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.port is not None:
        config.monitor_port = args.port

    root = Path(__file__).resolve().parents[1]
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = pick_device()
    retriever_config = to_retriever_config(config)

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
    print(f"补上的缺失小时：{summary['filled_missing_hours']}")

    run_dir = root / config.run_root / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    summary["device"] = str(device)
    summary["run_dir"] = str(run_dir)
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

    print("按未来相似度构造正例…")
    train_pools = build_positives(
        train, database, config.future_temperature, config.positive_future_quantile, config.positive_min_similarity
    )
    val_pools = build_positives(
        val, database, config.future_temperature, config.positive_future_quantile, config.positive_min_similarity
    )
    report = positive_report(train_pools)
    print(f"训练查询平均有正例 {report['mean_positives']} 条")

    model = RetrieverEncoder(retriever_config).to(device)
    head = [parameter for name, parameter in model.named_parameters() if parameter.requires_grad and name.startswith("output.")]
    adapters = [parameter for name, parameter in model.named_parameters() if parameter.requires_grad and not name.startswith("output.")]
    print(
        f"LoRA 检索器，投影 {sum(item.numel() for item in head)}，"
        f"适配器 {sum(item.numel() for item in adapters)}"
    )
    if not head and not adapters:
        raise RuntimeError("没有可训练参数。检查 peft 是否装好，以及 RetrieverEncoder 是否打开了 LoRA。")
    groups = []
    if head:
        groups.append({"params": head, "lr": config.learning_rate})
    if adapters:
        groups.append({"params": adapters, "lr": config.adapter_learning_rate})
    optimizer = torch.optim.AdamW(groups, weight_decay=config.weight_decay)
    train_rng = np.random.default_rng(config.seed)
    best_val = -float("inf")
    best_epoch = 0
    step = 0
    try:
        for epoch in range(1, config.epochs + 1):
            records = run_epoch(model, train, database, train_pools, config, device, train_rng, optimizer)
            for loss in records:
                step += 1
                append_metric(metrics_path, {
                    "kind": "step",
                    "epoch": epoch,
                    "step": step,
                    "loss": round(loss, 6),
                })
            val_rng = np.random.default_rng(config.seed)
            with torch.no_grad():
                val_records = run_epoch(model, val, database, val_pools, config, device, val_rng, None)
            train_loss = float(np.mean(records)) if records else float("nan")
            val_loss = float(np.mean(val_records)) if val_records else float("nan")
            val_retrieval = evaluate_retrieval(model, database, val, retriever_config, device)
            val_score = val_retrieval["retrieved_future_similarity"]
            append_metric(metrics_path, {
                "kind": "epoch",
                "epoch": epoch,
                "step": step,
                "train_loss": None if np.isnan(train_loss) else round(train_loss, 6),
                "val_loss": None if np.isnan(val_loss) else round(val_loss, 6),
                "val_retrieved_future_similarity": val_score,
                "val_hard_negative_rate": val_retrieval["hard_negative_rate"],
            })
            print(
                f"第 {epoch:02d} 轮  训练 {train_loss:.4f}  验证 {val_loss:.4f}  "
                f"检索 {val_score:.4f}"
            )
            save_model(run_dir / "encoder_last.pt", model, config)
            if val_score > best_val:
                best_val = val_score
                best_epoch = epoch
                save_model(run_dir / "encoder.pt", model, config)

        print("在测试查询上评估检索质量…")
        checkpoint = torch.load(run_dir / "encoder.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        result = evaluate_retrieval(model, database, test, retriever_config, device)
        append_metric(metrics_path, result)
        print(
            f"Top-{result['top_k']} 未来相似度  "
            f"检索器 {result['retrieved_future_similarity']:.4f}  "
            f"随机 {result['random_future_similarity']:.4f}  "
            f"Oracle {result['oracle_future_similarity']:.4f}  "
            f"难负例占比 {result['hard_negative_rate']}"
        )
        print(f"最好的模型：第 {best_epoch} 轮，验证检索 {best_val:.4f}，{run_dir / 'encoder.pt'}")
    except KeyboardInterrupt:
        save_model(run_dir / "encoder_last.pt", model, config)
        print("\n训练已中断，当前模型写在 encoder_last.pt。")
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
