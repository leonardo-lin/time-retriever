"""正例门槛、年份切分、验证/测试随机划分、MOMENT 补齐、生成器前向，以及正样本 LoRA 配置自检。"""

import inspect

import numpy as np
import torch

from chronos2.compare import model_ready, stack_point_forecast
from generator.compare import resolve_top_k
from generator.config import Config as GeneratorConfig
from generator.evaluate import forecast_metrics, sort_by_true
from generator.model import ForecastGenerator
from generator.retrieve import empty_retrieved, gather_cases, retrieve_oracle
from positive_lora.config import Config as PositiveLoraConfig
from positive_lora.config import to_retriever_config
from positive_lora.positives import pick_positives
from retriever.compare import page_payload
from retriever.config import Config
from retriever.data import WindowSet, _origins, pacific_midnight, random_eval_split
from retriever.evaluate import score_retrieval
from retriever.model import pad_to_moment
from retriever.sampling import classify_candidates


def test_positive_needs_top_tail_and_floor():
    config = Config(positive_future_quantile=0.95, positive_min_similarity=0.75)
    future = np.linspace(0.0, 1.0, 100)
    past = np.zeros(100)
    pos, _, _ = classify_candidates(past, future, config)
    assert future[pos].min() >= 0.75
    assert future[pos].min() >= np.quantile(future, 0.95)


def test_positive_falls_back_to_argmax():
    config = Config(positive_future_quantile=0.95, positive_min_similarity=0.99)
    future = np.linspace(0.0, 0.5, 100)
    pos, _, _ = classify_candidates(np.zeros(100), future, config)
    assert list(pos) == [int(np.argmax(future))]


def test_train_origins_stay_in_2025_eval_starts_in_2026():
    times = np.arange(np.datetime64("2024-12-01T00", "h"), np.datetime64("2026-03-01T00", "h"))
    train_from = pacific_midnight("2025-01-01")
    eval_from = pacific_midnight("2026-01-01")
    train = _origins(
        times,
        history=24,
        horizon=24,
        stride=1,
        origin_from=train_from,
        origin_before=eval_from,
    )
    eval_origins = _origins(times, history=24, horizon=24, stride=1, origin_from=eval_from)
    assert times[train[0]] >= train_from
    assert times[train[-1]] < eval_from
    assert times[eval_origins[0]] >= eval_from


def test_random_eval_split_uses_all_indices_and_is_not_chronological():
    val, test = random_eval_split(20, 0.5, 0.5, seed=42)
    assert len(val) == 10 and len(test) == 10
    assert set(val.tolist()) | set(test.tolist()) == set(range(20))
    assert set(val.tolist()) & set(test.tolist()) == set()
    assert not np.array_equal(val, np.arange(10))


def test_pad_to_moment_masks_left_pad_only():
    windows = torch.arange(504, dtype=torch.float32).unsqueeze(0)
    x_enc, mask = pad_to_moment(windows)
    assert tuple(x_enc.shape) == (1, 1, 512)
    assert tuple(mask.shape) == (1, 512)
    assert torch.equal(mask[0, :8], torch.zeros(8))
    assert torch.equal(mask[0, 8:], torch.ones(504))
    assert torch.equal(x_enc[0, 0, :8], torch.zeros(8))
    assert torch.equal(x_enc[0, 0, 8:], windows[0])


def test_generator_default_backbone_is_local_moment():
    config = GeneratorConfig()
    assert config.backbone == "moment"
    assert config.moment_path == "models/MOMENT-1-large"
    assert config.freeze_backbone is True


def test_generator_forward_has_no_current_y_and_handles_k0():
    config = GeneratorConfig(backbone="scratch")
    model = ForecastGenerator(config, history_hours=504, horizon_hours=168)
    names = inspect.signature(model.forward).parameters
    assert "y" not in names
    x = torch.arange(504, dtype=torch.float32).unsqueeze(0) + 1000
    xi = torch.zeros(1, 0, 504)
    yi = torch.zeros(1, 0, 168)
    y_hat, center, scale = model(x, xi, yi)
    assert tuple(y_hat.shape) == (1, 168)
    assert torch.allclose(center.squeeze(), x.mean())
    assert torch.allclose(scale.squeeze(), x.std(unbiased=False), atol=1e-5)


def test_generator_forward_accepts_retrieved_cases():
    config = GeneratorConfig(backbone="scratch")
    model = ForecastGenerator(config, history_hours=504, horizon_hours=168)
    x = torch.randn(2, 504)
    xi = torch.randn(2, 5, 504)
    yi = torch.randn(2, 5, 168)
    y_hat, _, _ = model(x, xi, yi)
    assert tuple(y_hat.shape) == (2, 168)


def test_retrieve_oracle_ranks_by_future_not_past():
    database = WindowSet(
        name="database",
        x=np.array([[0.0, 1.0], [9.0, 10.0], [1.0, 0.0]], dtype=np.float32),
        y=np.array([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0], [1.0, 2.0, 2.0, 1.0]], dtype=np.float32),
        origin_utc=np.array(["2024-01-01", "2024-01-02", "2024-01-03"], dtype="datetime64[h]"),
        future_end_utc=np.array(["2024-01-08", "2024-01-09", "2024-01-10"], dtype="datetime64[h]"),
    )
    queries = WindowSet(
        name="test",
        x=np.array([[9.0, 10.0]], dtype=np.float32),
        y=np.array([[4.0, 3.0, 2.0, 1.0]], dtype=np.float32),
        origin_utc=np.array(["2026-01-01"], dtype="datetime64[h]"),
        future_end_utc=np.array(["2026-01-08"], dtype="datetime64[h]"),
    )
    hit = retrieve_oracle(queries, database, k=2, temperature=1.0)
    assert hit.indices.shape == (1, 2)
    assert int(hit.indices[0, 0]) == 1
    assert int(hit.indices[0, 1]) == 2


def test_gather_cases_keeps_empty_k():
    database = WindowSet(
        name="database",
        x=np.arange(20, dtype=np.float32).reshape(2, 10),
        y=np.arange(8, dtype=np.float32).reshape(2, 4),
        origin_utc=np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[h]"),
        future_end_utc=np.array(["2024-01-08", "2024-01-09"], dtype="datetime64[h]"),
    )
    retrieved = empty_retrieved(3)
    xi, yi = gather_cases(database, retrieved.indices)
    assert xi.shape == (3, 0, 10)
    assert yi.shape == (3, 0, 4)


def test_positive_lora_picks_top_tail_and_floor():
    future = np.linspace(0.0, 1.0, 100)
    pos = pick_positives(future, 0.95, 0.75)
    assert future[pos].min() >= 0.75
    assert future[pos].min() >= np.quantile(future, 0.95)


def test_positive_lora_falls_back_to_argmax():
    future = np.linspace(0.0, 0.5, 100)
    pos = pick_positives(future, 0.95, 0.99)
    assert list(pos) == [int(np.argmax(future))]


def test_positive_lora_locks_frozen_moment_lora():
    mapped = to_retriever_config(PositiveLoraConfig())
    assert mapped.encoder == "moment"
    assert mapped.freeze_backbone is True
    assert mapped.use_lora is True


def test_sort_by_true_keeps_pred_alignment():
    y_true = np.array([[3.0, 1.0], [2.0, 0.0]], dtype=np.float32)
    y_hat = np.array([[30.0, 10.0], [20.0, 0.0]], dtype=np.float32)
    origins = np.array(["2026-01-01T00", "2026-01-02T00"], dtype="datetime64[h]")
    ranked = sort_by_true(y_true, y_hat, origins)
    assert list(ranked["true"]) == [0.0, 1.0, 2.0, 3.0]
    assert list(ranked["pred"]) == [0.0, 10.0, 20.0, 30.0]
    assert list(ranked["query"]) == [1, 0, 1, 0]
    assert list(ranked["hour"]) == [1, 1, 0, 0]
    assert str(ranked["origin_utc"][0]) == "2026-01-02T00"


def test_score_retrieval_means_and_hard_negative_rate():
    config = Config(top_k=2, high_past_quantile=0.50, low_future_quantile=0.50)
    future = np.array([
        [0.10, 0.90, 0.80, 0.20],
        [0.15, 0.25, 0.95, 0.40],
    ], dtype=np.float32)
    past = np.array([
        [0.95, 0.10, 0.20, 0.90],
        [0.90, 0.85, 0.10, 0.20],
    ], dtype=np.float32)
    retrieved = np.array([[1, 2], [2, 3]])
    oracle = np.array([[1, 2], [2, 3]])
    random = np.array([[0, 3], [0, 1]])
    report = score_retrieval(retrieved, oracle, random, future, past, config)
    summary = report["summary"]
    assert summary["retrieved_future_similarity"] == round(float(np.mean([0.90, 0.80, 0.95, 0.40])), 4)
    assert summary["oracle_future_similarity"] == summary["retrieved_future_similarity"]
    assert summary["random_future_similarity"] == round(float(np.mean([0.10, 0.20, 0.15, 0.25])), 4)
    assert summary["oracle_future_similarity"] > summary["random_future_similarity"]
    assert summary["hard_negative_rate"] == 0.0
    assert summary["random_hard_negative_rate"] == 1.0
    assert summary["gap_closed"] == 1.0


def test_compare_payload_keeps_three_baselines():
    config = Config(top_k=2, high_past_quantile=0.50, low_future_quantile=0.50)
    future = np.array([
        [0.10, 0.90, 0.80, 0.20],
        [0.15, 0.25, 0.95, 0.40],
    ], dtype=np.float32)
    past = np.array([
        [0.95, 0.10, 0.20, 0.90],
        [0.90, 0.85, 0.10, 0.20],
    ], dtype=np.float32)
    retrieved = np.array([[1, 2], [2, 3]])
    ranked = score_retrieval(
        retrieved,
        np.array([[1, 2], [2, 3]]),
        np.array([[0, 3], [0, 1]]),
        future,
        past,
        config,
    )
    database = WindowSet(
        name="database",
        x=np.ones((4, 8), dtype=np.float32),
        y=np.arange(16, dtype=np.float32).reshape(4, 4),
        origin_utc=np.array(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"], dtype="datetime64[h]"),
        future_end_utc=np.array(["2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11"], dtype="datetime64[h]"),
    )
    queries = WindowSet(
        name="test",
        x=np.ones((2, 8), dtype=np.float32),
        y=np.ones((2, 4), dtype=np.float32),
        origin_utc=np.array(["2026-01-01", "2026-01-02"], dtype="datetime64[h]"),
        future_end_utc=np.array(["2026-01-08", "2026-01-09"], dtype="datetime64[h]"),
    )
    payload = page_payload(ranked, database, queries)
    assert payload["retrieved"] == ranked["summary"]["retrieved_future_similarity"]
    assert payload["random"] == ranked["summary"]["random_future_similarity"]
    assert payload["oracle"] == ranked["summary"]["oracle_future_similarity"]
    assert payload["hard_negative_rate"] == 0.0
    assert len(payload["retrieved_sorted"]) == 2
    assert payload["oracle_sorted"] == sorted(payload["oracle_sorted"], reverse=True)
    assert payload["examples"]
    assert payload["hard_points"] or payload["retrieved_points"]


def test_compare_uses_checkpoint_top_k_not_current_default():
    assert resolve_top_k({"top_k": 5}, "frozen", None) == 5
    assert resolve_top_k({"top_k": 2}, "frozen", None) == 2
    assert resolve_top_k({"top_k": 2}, "oracle", None) == 2
    assert resolve_top_k({"top_k": 5}, "none", None) == 0
    assert resolve_top_k({"top_k": 5}, "frozen", 2) == 2
    try:
        resolve_top_k({}, "frozen", None)
    except RuntimeError:
        return
    raise AssertionError("missing top_k should stop")


def test_chronos2_point_forecast_takes_first_variate():
    y_hat = stack_point_forecast([
        np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        np.array([[4.0, 5.0, 6.0]], dtype=np.float32),
    ])
    assert y_hat.shape == (2, 3)
    assert list(y_hat[0]) == [1.0, 2.0, 3.0]
    assert list(y_hat[1]) == [4.0, 5.0, 6.0]


def test_chronos2_model_ready_needs_weights():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder)
        (path / "config.json").write_text("{}", encoding="utf-8")
        assert not model_ready(path)
        (path / "model.safetensors").write_bytes(b"x")
        assert model_ready(path)


def test_forecast_metrics_split_by_day():
    y = np.zeros((2, 168), dtype=np.float32)
    y_hat = np.zeros((2, 168), dtype=np.float32)
    y_hat[:, :24] = 2.0
    result = forecast_metrics(y, y_hat, patch_hours=24)
    assert result["day_mae"][0] == 2.0
    assert result["day_mae"][1] == 0.0
    assert result["day_rmse"][0] == 2.0
    assert result["day_rmse"][1] == 0.0
    assert len(result["day_mae"]) == 7
    assert len(result["day_rmse"]) == 7
    assert result["mae"] == round(2.0 * 24 / 168, 4)


if __name__ == "__main__":
    test_positive_needs_top_tail_and_floor()
    test_positive_falls_back_to_argmax()
    test_train_origins_stay_in_2025_eval_starts_in_2026()
    test_random_eval_split_uses_all_indices_and_is_not_chronological()
    test_pad_to_moment_masks_left_pad_only()
    test_generator_default_backbone_is_local_moment()
    test_generator_forward_has_no_current_y_and_handles_k0()
    test_generator_forward_accepts_retrieved_cases()
    test_gather_cases_keeps_empty_k()
    test_retrieve_oracle_ranks_by_future_not_past()
    test_sort_by_true_keeps_pred_alignment()
    test_score_retrieval_means_and_hard_negative_rate()
    test_compare_payload_keeps_three_baselines()
    test_compare_uses_checkpoint_top_k_not_current_default()
    test_forecast_metrics_split_by_day()
    test_chronos2_point_forecast_takes_first_variate()
    test_chronos2_model_ready_needs_weights()
    test_positive_lora_picks_top_tail_and_floor()
    test_positive_lora_falls_back_to_argmax()
    test_positive_lora_locks_frozen_moment_lora()
    print("ok")

