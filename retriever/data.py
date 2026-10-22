"""读取 CAISO 小时负荷，并切成检索库与查询。"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from retriever.config import Config

PACIFIC = ZoneInfo("America/Los_Angeles")


@dataclass
class WindowSet:
    """一组预测案例。origin 是 X 的最后一个小时，也就是做预测的“现在”。"""

    name: str
    x: np.ndarray
    y: np.ndarray
    origin_utc: np.ndarray
    future_end_utc: np.ndarray

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def subset(self, indices: np.ndarray) -> "WindowSet":
        return WindowSet(
            name=self.name,
            x=self.x[indices],
            y=self.y[indices],
            origin_utc=self.origin_utc[indices],
            future_end_utc=self.future_end_utc[indices],
        )


def pacific_midnight(day: str) -> np.datetime64:
    local = datetime.fromisoformat(day).replace(tzinfo=PACIFIC)
    utc = local.astimezone(timezone.utc).replace(tzinfo=None)
    return np.datetime64(utc.strftime("%Y-%m-%dT%H"), "h")


def format_pacific(stamp: np.datetime64) -> str:
    seconds = int(stamp.astype("datetime64[s]").astype(np.int64))
    aware = datetime.fromtimestamp(seconds, tz=timezone.utc).astimezone(PACIFIC)
    return aware.strftime("%Y-%m-%d %H:%M %Z")


def load_hourly_load(csv_path: Path, tac_area: str) -> tuple[np.ndarray, np.ndarray, int]:
    """返回完整小时网格上的负荷。中间缺的小时用线性插值补上。"""
    stamps: list[np.datetime64] = []
    values: list[float] = []
    with csv_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["TAC_AREA_NAME"] != tac_area:
                continue
            parsed = datetime.fromisoformat(row["INTERVALSTARTTIME_GMT"]).astimezone(timezone.utc)
            stamps.append(np.datetime64(parsed.strftime("%Y-%m-%dT%H"), "h"))
            values.append(float(row["MW"]))
    if not stamps:
        raise RuntimeError(f"在 {csv_path} 里没有找到 {tac_area}")

    order = np.argsort(np.array(stamps), kind="mergesort")
    observed_time = np.array(stamps)[order]
    observed_value = np.asarray(values, dtype=np.float64)[order]
    # 同一小时若重复，取平均。
    unique_time, inverse = np.unique(observed_time, return_inverse=True)
    summed = np.zeros(len(unique_time), dtype=np.float64)
    counts = np.zeros(len(unique_time), dtype=np.float64)
    np.add.at(summed, inverse, observed_value)
    np.add.at(counts, inverse, 1.0)
    observed_value = summed / counts

    grid = np.arange(unique_time[0], unique_time[-1] + np.timedelta64(1, "h"), dtype="datetime64[h]")
    series = np.full(len(grid), np.nan, dtype=np.float64)
    positions = np.searchsorted(grid, unique_time)
    series[positions] = observed_value
    missing = np.isnan(series)
    if missing.any():
        known = np.flatnonzero(~missing)
        series[missing] = np.interp(np.flatnonzero(missing), known, series[known])
    return grid, series.astype(np.float32), int(missing.sum())


def _gather(series: np.ndarray, origins: np.ndarray, history: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.empty((len(origins), history), dtype=np.float32)
    y = np.empty((len(origins), horizon), dtype=np.float32)
    for row, origin in enumerate(origins):
        x[row] = series[origin - history + 1 : origin + 1]
        y[row] = series[origin + 1 : origin + 1 + horizon]
    return x, y


def _pacific_hour(stamp: np.datetime64) -> int:
    seconds = int(stamp.astype("datetime64[s]").astype(np.int64))
    return datetime.fromtimestamp(seconds, tz=timezone.utc).astimezone(PACIFIC).hour


def _candidate_indices(times: np.ndarray, first: int, last: int, stride: int) -> np.ndarray:
    if stride == 24:
        # 固定在本地时间 23:00，避免夏令时把“每天一条”挪到不同钟点。
        return np.array(
            [index for index in range(first, last + 1) if _pacific_hour(times[index]) == 23],
            dtype=np.int64,
        )
    return np.arange(first, last + 1, stride, dtype=np.int64)


def _origins(
    times: np.ndarray,
    history: int,
    horizon: int,
    stride: int,
    *,
    origin_from: np.datetime64 | None = None,
    origin_before: np.datetime64 | None = None,
    future_end_before: np.datetime64 | None = None,
) -> np.ndarray:
    first = history - 1
    last = len(times) - 1 - horizon
    picked: list[int] = []
    for origin in _candidate_indices(times, first, last, stride):
        if origin_from is not None and times[origin] < origin_from:
            continue
        if origin_before is not None and times[origin] >= origin_before:
            continue
        future_end = times[int(origin) + horizon]
        if future_end_before is not None and future_end >= future_end_before:
            continue
        picked.append(int(origin))
    if not picked:
        raise RuntimeError("这个时间范围里没有切出任何窗口，请检查日期或步长。")
    return np.asarray(picked, dtype=np.int64)


def make_windows(
    times: np.ndarray,
    series: np.ndarray,
    origins: np.ndarray,
    history: int,
    horizon: int,
    name: str,
) -> WindowSet:
    x, y = _gather(series, origins, history, horizon)
    return WindowSet(
        name=name,
        x=x,
        y=y,
        origin_utc=times[origins],
        future_end_utc=times[origins + horizon],
    )


def random_eval_split(
    count: int,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if val_ratio <= 0 or test_ratio <= 0:
        raise RuntimeError("验证和测试比例必须大于 0。")
    n_val = int(round(count * val_ratio / (val_ratio + test_ratio)))
    n_test = count - n_val
    if min(n_val, n_test) < 1:
        raise RuntimeError(f"2026 年查询只有 {count} 条，不够切验证和测试。")
    order = np.random.default_rng(seed).permutation(count)
    return order[:n_val], order[n_val:]


def load_experiment(config: Config, root: Path) -> tuple[WindowSet, WindowSet, WindowSet, WindowSet, dict]:
    csv_path = root / config.csv_path
    times, series, filled_hours = load_hourly_load(csv_path, config.tac_area)
    history = config.history_hours
    horizon = config.horizon_hours
    if history % config.patch_hours != 0:
        raise RuntimeError("history_hours 必须能被 patch_hours 整除。")

    database_before = pacific_midnight(config.database_before)
    train_from = pacific_midnight(config.train_from)
    eval_from = pacific_midnight(config.eval_from)
    db_origins = _origins(
        times, history, horizon, config.stride_hours, future_end_before=database_before
    )
    train_origins = _origins(
        times, history, horizon, config.stride_hours, origin_from=train_from, origin_before=eval_from
    )
    eval_origins = _origins(
        times, history, horizon, config.stride_hours, origin_from=eval_from
    )
    database = make_windows(times, series, db_origins, history, horizon, "database")
    train = make_windows(times, series, train_origins, history, horizon, "train")
    eval_queries = make_windows(times, series, eval_origins, history, horizon, "eval")

    if database.future_end_utc.max() >= train.origin_utc.min():
        raise RuntimeError("检索库的未来和查询时段重叠，存在泄漏。")

    val_idx, test_idx = random_eval_split(
        len(eval_queries), config.val_ratio, config.test_ratio, config.seed
    )
    val = eval_queries.subset(val_idx)
    val.name = "val"
    test = eval_queries.subset(test_idx)
    test.name = "test"

    gap_hours = int((train.origin_utc.min() - database.future_end_utc.max()) / np.timedelta64(1, "h"))
    summary = {
        "tac_area": config.tac_area,
        "filled_missing_hours": filled_hours,
        "history_hours": history,
        "horizon_hours": horizon,
        "stride_hours": config.stride_hours,
        "database_cases": len(database),
        "database_origin_start": format_pacific(database.origin_utc.min()),
        "database_origin_end": format_pacific(database.origin_utc.max()),
        "database_future_end": format_pacific(database.future_end_utc.max()),
        "train_cases": len(train),
        "train_start": format_pacific(train.origin_utc.min()),
        "train_end": format_pacific(train.origin_utc.max()),
        "val_cases": len(val),
        "val_start": format_pacific(val.origin_utc.min()),
        "val_end": format_pacific(val.origin_utc.max()),
        "test_cases": len(test),
        "test_start": format_pacific(test.origin_utc.min()),
        "test_end": format_pacific(test.origin_utc.max()),
        "leakage_gap_hours": gap_hours,
        "series_start": format_pacific(times[0]),
        "series_end": format_pacific(times[-1]),
    }
    return database, train, val, test, summary
