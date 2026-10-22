"""训练未来感知检索器，并在本地网页上实时画损失。

在项目根目录执行：

    ./venv/bin/python -m retriever.train
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

from retriever.config import Config
from retriever.data import WindowSet, load_experiment
from retriever.evaluate import evaluate_retrieval
from retriever.loss import retriever_losses
from retriever.model import RetrieverEncoder
from retriever.monitor import start_monitor
from retriever.sampling import Pools, build_pools, pool_report


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def take(pool: np.ndarray, count: int, rng: np.random.Generator) -> list[int]:
    if count <= 0 or len(pool) == 0:
        return []
    picked = rng.choice(pool, size=count, replace=len(pool) < count)
    return [int(item) for item in np.atleast_1d(picked)]


def sample_candidates(pools: Pools, index: int, rng: np.random.Generator, n_hard: int, n_easy: int):
    positive = int(rng.choice(pools.positive[index]))
    hard = take(pools.hard[index], n_hard, rng)
    easy = take(pools.easy[index], n_easy, rng)
    chosen = hard + easy
    target = n_hard + n_easy
    if len(chosen) < target:
        backup = pools.easy[index] if len(pools.easy[index]) else pools.hard[index]
        chosen.extend(take(backup, target - len(chosen), rng))
    if len(chosen) < target:
        return None
    return positive, chosen[:target]


def encode(model: RetrieverEncoder, windows: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(windows)).to(device)
    return model(tensor)


def batch_losses(
    model: RetrieverEncoder,
    queries: WindowSet,
    database: WindowSet,
    pools: Pools,
    indices: np.ndarray,
    rng: np.random.Generator,
    config: Config,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    positives = []
    negatives = []
    kept = []
    per_query = config.hard_negatives + config.easy_negatives
    for index in indices:
        sampled = sample_candidates(pools, int(index), rng, config.hard_negatives, config.easy_negatives)
        if sampled is None:
            continue
        positive, negative = sampled
        kept.append(int(index))
        positives.append(positive)
        negatives.append(negative)
    if not kept:
        return None
    negative_windows = database.x[np.array(negatives)].reshape(len(kept) * per_query, -1)
    packed = np.concatenate([
        queries.x[np.array(kept)],
        database.x[np.array(positives)],
        negative_windows,
    ])
    encoded = encode(model, packed, device)
    query_z = encoded[: len(kept)]
    positive_z = encoded[len(kept) : 2 * len(kept)]
    negative_z = encoded[2 * len(kept) :].reshape(len(kept), per_query, -1)
    contrastive, ranking = retriever_losses(
        query_z,
        positive_z,
        negative_z,
        config.contrastive_temperature,
        config.margin,
    )
    total = config.lambda_contrastive * contrastive + config.lambda_ranking * ranking
    return total, contrastive, ranking


def run_epoch(
    model: RetrieverEncoder,
    queries: WindowSet,
    database: WindowSet,
    pools: Pools,
    config: Config,
    device: torch.device,
    rng: np.random.Generator,
    optimizer: torch.optim.Optimizer | None,
) -> list[tuple[float, float, float]]:
    training = optimizer is not None
    model.train(training)
    order = rng.permutation(len(queries))
    recorded = []
    for start in range(0, len(order), config.batch_size):
        indices = order[start : start + config.batch_size]
        losses = batch_losses(model, queries, database, pools, indices, rng, config, device)
        if losses is None:
            continue
        total, contrastive, ranking = losses
        if training:
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
        recorded.append((float(total.detach()), float(contrastive.detach()), float(ranking.detach())))
    return recorded


def append_metric(path: Path, record: dict) -> None:
    record["time"] = datetime.now().isoformat(timespec="seconds")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_model(path: Path, model: RetrieverEncoder, config: Config) -> None:
    torch.save({"model": model.state_dict(), "config": config.to_dict()}, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练未来感知负荷检索器")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--stride-hours", type=int)
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
    if args.stride_hours is not None:
        config.stride_hours = args.stride_hours
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.port is not None:
        config.monitor_port = args.port

    root = Path(__file__).resolve().parents[1]
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = pick_device()

    print("读取负荷并切分窗口…")
    database, train, val, test, summary = load_experiment(config, root)
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

    print("按未来相似度构造正例、难负例、易负例…")
    train_pools = build_pools(train, database, config)
    val_pools = build_pools(val, database, config)
    report = pool_report(train_pools)
    print(
        f"训练查询平均有正例 {report['mean_positives']} 条、"
        f"难负例 {report['mean_hard_negatives']} 条、"
        f"易负例 {report['mean_easy_negatives']} 条；"
        f"{report['queries_with_hard_negatives']}/{report['queries']} 条查询找得到难负例"
    )

    model = RetrieverEncoder(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    train_rng = np.random.default_rng(config.seed)
    best_val = -float("inf")
    best_epoch = 0
    step = 0
    try:
        for epoch in range(1, config.epochs + 1):
            records = run_epoch(model, train, database, train_pools, config, device, train_rng, optimizer)
            for total, contrastive, ranking in records:
                step += 1
                append_metric(metrics_path, {
                    "kind": "step",
                    "epoch": epoch,
                    "step": step,
                    "loss": round(total, 6),
                    "contrastive": round(contrastive, 6),
                    "ranking": round(ranking, 6),
                })
            val_rng = np.random.default_rng(config.seed)
            with torch.no_grad():
                val_records = run_epoch(model, val, database, val_pools, config, device, val_rng, None)
            train_loss = float(np.mean([item[0] for item in records])) if records else float("nan")
            val_loss = float(np.mean([item[0] for item in val_records])) if val_records else float("nan")
            val_retrieval = evaluate_retrieval(model, database, val, config, device)
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
        result = evaluate_retrieval(model, database, test, config, device)
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
