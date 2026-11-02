"""用训好的检索器跑完全部测试查询，对照 Oracle、随机和难负例。

在项目根目录执行：

    ./venv/bin/python -m retriever.compare --encoder runs/retriever/<run>/encoder.pt
"""

from __future__ import annotations

import argparse
import json
import webbrowser
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

from retriever.config import Config
from retriever.data import format_pacific, load_experiment
from retriever.evaluate import rank_retrieval
from retriever.model import RetrieverEncoder
from retriever.sampling import zscore_rows


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_path(root: Path, path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def config_from_checkpoint(checkpoint: dict) -> Config:
    if "config" not in checkpoint:
        raise RuntimeError("encoder.pt 里没有 config。")
    allowed = {item.name for item in fields(Config)}
    payload = {key: value for key, value in checkpoint["config"].items() if key in allowed}
    return Config(**payload)


def rounded(values: np.ndarray, digits: int = 4) -> list[float]:
    return [round(float(item), digits) for item in values]


def mean_shape(windows: np.ndarray) -> list[float]:
    zscored = zscore_rows(windows)
    return rounded(zscored.mean(axis=0), 3)


def scatter_points(index: np.ndarray, past: np.ndarray, future: np.ndarray, hard_sets: list[set[int]] | None = None) -> tuple[list[dict], list[dict]]:
    ordinary: list[dict] = []
    hard: list[dict] = []
    for row, chosen in enumerate(index):
        hard_set = hard_sets[row] if hard_sets is not None else set()
        for item in chosen:
            point = {
                "past": round(float(past[row, item]), 4),
                "future": round(float(future[row, item]), 4),
            }
            if item in hard_set:
                hard.append(point)
            else:
                ordinary.append(point)
    return ordinary, hard


def pick_example_rows(scores: np.ndarray) -> list[int]:
    if len(scores) == 0:
        return []
    order = np.argsort(scores)
    picks = [int(order[0]), int(order[len(order) // 2]), int(order[-1])]
    seen: set[int] = set()
    unique: list[int] = []
    for row in picks:
        if row not in seen:
            seen.add(row)
            unique.append(row)
    return unique


def example_payload(ranked: dict, database, test, rows: list[int]) -> list[dict]:
    labels = ["Worst retrieval", "Median retrieval", "Best retrieval"]
    if len(rows) != 3:
        labels = [f"Query {row}" for row in rows]
    examples = []
    for label, row in zip(labels, rows):
        true = zscore_rows(test.y[row : row + 1])[0]
        examples.append({
            "title": label,
            "origin": format_pacific(test.origin_utc[row]),
            "retrieved": round(float(ranked["retrieved_per_query"][row]), 4),
            "random": round(float(ranked["random_per_query"][row]), 4),
            "oracle": round(float(ranked["oracle_per_query"][row]), 4),
            "true": rounded(true, 3),
            "retrieved_y": mean_shape(database.y[ranked["retrieved"][row]]),
            "random_y": mean_shape(database.y[ranked["random"][row]]),
            "oracle_y": mean_shape(database.y[ranked["oracle"][row]]),
        })
    return examples


def page_payload(ranked: dict, database, test) -> dict:
    summary = ranked["summary"]
    order = np.argsort(-ranked["oracle_per_query"])
    retrieved_points, hard_points = scatter_points(
        ranked["retrieved"],
        ranked["past_similarity"],
        ranked["future_similarity"],
        ranked["hard_sets"],
    )
    oracle_points, _ = scatter_points(
        ranked["oracle"],
        ranked["past_similarity"],
        ranked["future_similarity"],
    )
    return {
        "queries": summary["test_queries"],
        "top_k": summary["top_k"],
        "retrieved": summary["retrieved_future_similarity"],
        "random": summary["random_future_similarity"],
        "oracle": summary["oracle_future_similarity"],
        "hard_negative_rate": summary["hard_negative_rate"],
        "oracle_hard_negative_rate": summary["oracle_hard_negative_rate"],
        "random_hard_negative_rate": summary["random_hard_negative_rate"],
        "gap_closed": summary["gap_closed"],
        "retrieved_sorted": rounded(ranked["retrieved_per_query"][order]),
        "random_sorted": rounded(ranked["random_per_query"][order]),
        "oracle_sorted": rounded(ranked["oracle_per_query"][order]),
        "retrieved_points": retrieved_points,
        "hard_points": hard_points,
        "oracle_points": oracle_points,
        "examples": example_payload(ranked, database, test, pick_example_rows(ranked["retrieved_per_query"])),
    }


def write_html(path: Path, payload: dict) -> None:
    template = (Path(__file__).resolve().parent / "compare.html").read_text(encoding="utf-8")
    path.write_text(template.replace("__DATA__", json.dumps(payload, ensure_ascii=False)), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对照测试检索的 Oracle、随机和难负例")
    parser.add_argument("--encoder", required=True, help="检索器 encoder.pt 路径")
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    encoder_path = resolve_path(root, args.encoder)
    if not encoder_path.is_file():
        raise RuntimeError(f"找不到检索器权重：{encoder_path}")

    device = pick_device()
    stored = torch.load(encoder_path, map_location=device, weights_only=False)
    if not isinstance(stored, dict) or "model" not in stored:
        raise RuntimeError(f"{encoder_path} 里没有 model 权重。")
    config = config_from_checkpoint(stored)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    print("读取负荷并切分窗口…")
    database, _train, _val, test, summary = load_experiment(config, root)
    print(f"测试 {summary['test_cases']} 条：{summary['test_start']} → {summary['test_end']}")

    model = RetrieverEncoder(config).to(device)
    model.load_state_dict(stored["model"])
    print("在测试查询上评估检索质量…")
    ranked = rank_retrieval(model, database, test, config, device)
    result = ranked["summary"]
    payload = page_payload(ranked, database, test)

    html_path = encoder_path.parent / "test_retrieval.html"
    write_html(html_path, payload)
    print(
        f"Top-{result['top_k']} 未来相似度  "
        f"检索器 {result['retrieved_future_similarity']:.4f}  "
        f"随机 {result['random_future_similarity']:.4f}  "
        f"Oracle {result['oracle_future_similarity']:.4f}  "
        f"难负例占比 {result['hard_negative_rate']}  "
        f"已追回 {result['gap_closed']}"
    )
    print(f"对照页：{html_path}")
    if not args.no_browser:
        webbrowser.open(html_path.resolve().as_uri())


if __name__ == "__main__":
    main()
