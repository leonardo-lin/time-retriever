"""反标准化之后的预测误差。"""

from __future__ import annotations

import numpy as np
import torch

from generator.config import Config
from generator.model import ForecastGenerator
from generator.retrieve import Retrieved, gather_cases
from retriever.data import WindowSet


def sort_by_true(y_true: np.ndarray, y_hat: np.ndarray, origin_utc: np.ndarray) -> dict[str, np.ndarray]:
    """把所有测试小时按真实值从低到高排，预测跟着同一套下标走。"""
    n_queries, horizon = y_true.shape
    true_flat = np.ascontiguousarray(y_true).reshape(-1)
    pred_flat = np.ascontiguousarray(y_hat).reshape(-1)
    query = np.repeat(np.arange(n_queries), horizon)
    hour = np.tile(np.arange(horizon), n_queries)
    order = np.argsort(true_flat, kind="mergesort")
    query_ordered = query[order]
    return {
        "true": true_flat[order],
        "pred": pred_flat[order],
        "query": query_ordered,
        "hour": hour[order],
        "origin_utc": origin_utc[query_ordered],
    }


def forecast_metrics(y_true: np.ndarray, y_hat: np.ndarray, patch_hours: int) -> dict:
    err = y_hat - y_true
    mae = float(np.abs(err).mean())
    rmse = float(np.sqrt((err ** 2).mean()))
    mape = float((np.abs(err) / np.clip(np.abs(y_true), 1e-6, None)).mean() * 100.0)
    day_mae = []
    day_rmse = []
    horizon = y_true.shape[-1]
    for start in range(0, horizon, patch_hours):
        day_err = err[..., start : start + patch_hours]
        day_mae.append(round(float(np.abs(day_err).mean()), 4))
        day_rmse.append(round(float(np.sqrt((day_err ** 2).mean())), 4))
    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "mape": round(mape, 4),
        "day_mae": day_mae,
        "day_rmse": day_rmse,
    }


def predict_forecast(
    model: ForecastGenerator,
    queries: WindowSet,
    database: WindowSet,
    retrieved: Retrieved,
    config: Config,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(queries), config.batch_size):
            index = np.arange(start, min(start + config.batch_size, len(queries)))
            x = torch.from_numpy(np.ascontiguousarray(queries.x[index])).to(device)
            xi, yi = gather_cases(database, retrieved.indices[index])
            xi_t = torch.from_numpy(np.ascontiguousarray(xi)).to(device)
            yi_t = torch.from_numpy(np.ascontiguousarray(yi)).to(device)
            y_hat_norm, center, scale = model(x, xi_t, yi_t)
            preds.append((y_hat_norm * scale + center).cpu().numpy())
    return np.concatenate(preds, axis=0)


def evaluate_forecast(
    model: ForecastGenerator,
    queries: WindowSet,
    database: WindowSet,
    retrieved: Retrieved,
    config: Config,
    device: torch.device,
) -> dict:
    y_hat = predict_forecast(model, queries, database, retrieved, config, device)
    result = forecast_metrics(queries.y, y_hat, config.patch_hours)
    result["kind"] = "test"
    result["queries"] = len(queries)
    result["top_k"] = int(retrieved.indices.shape[1])
    return result
