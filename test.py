"""正例门槛、年份切分和验证/测试随机划分的自检。"""

import numpy as np

from retriever.config import Config
from retriever.data import _origins, pacific_midnight, random_eval_split
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


if __name__ == "__main__":
    test_positive_needs_top_tail_and_floor()
    test_positive_falls_back_to_argmax()
    test_train_origins_stay_in_2025_eval_starts_in_2026()
    test_random_eval_split_uses_all_indices_and_is_not_chronological()
    print("ok")
