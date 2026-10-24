"""正例门槛、年份切分、验证/测试随机划分、MOMENT 补齐，以及生成器前向的自检。"""

import inspect

import numpy as np
import torch

from generator.config import Config as GeneratorConfig
from generator.evaluate import forecast_metrics
from generator.model import ForecastGenerator
from generator.retrieve import empty_retrieved, gather_cases
from retriever.config import Config
from retriever.data import WindowSet, _origins, pacific_midnight, random_eval_split
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


def test_generator_forward_has_no_current_y_and_handles_k0():
    config = GeneratorConfig()
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
    config = GeneratorConfig()
    model = ForecastGenerator(config, history_hours=504, horizon_hours=168)
    x = torch.randn(2, 504)
    xi = torch.randn(2, 5, 504)
    yi = torch.randn(2, 5, 168)
    y_hat, _, _ = model(x, xi, yi)
    assert tuple(y_hat.shape) == (2, 168)


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


def test_forecast_metrics_split_by_day():
    y = np.zeros((2, 168), dtype=np.float32)
    y_hat = np.zeros((2, 168), dtype=np.float32)
    y_hat[:, :24] = 2.0
    result = forecast_metrics(y, y_hat, patch_hours=24)
    assert result["day_mae"][0] == 2.0
    assert result["day_mae"][1] == 0.0
    assert len(result["day_mae"]) == 7
    assert result["mae"] == round(2.0 * 24 / 168, 4)


if __name__ == "__main__":
    test_positive_needs_top_tail_and_floor()
    test_positive_falls_back_to_argmax()
    test_train_origins_stay_in_2025_eval_starts_in_2026()
    test_random_eval_split_uses_all_indices_and_is_not_chronological()
    test_pad_to_moment_masks_left_pad_only()
    test_generator_forward_has_no_current_y_and_handles_k0()
    test_generator_forward_accepts_retrieved_cases()
    test_gather_cases_keeps_empty_k()
    test_forecast_metrics_split_by_day()
    print("ok")

