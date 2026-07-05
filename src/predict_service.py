# -*- coding: utf-8 -*-
"""
predict_service.py — Dịch vụ dự đoán dùng chung cho web demo (và có thể tái dùng).

Cho một NGÀY bất kỳ, hàm predict_grid_for_date() sẽ:
  1. phủ lưới ô đều lên Việt Nam
  2. lấy thời tiết + độ cao cho tâm mỗi ô vào ngày đó (có cache)
  3. dùng mô hình đã train để dự đoán XÁC SUẤT nguy cơ cháy cho từng ô
  4. trả về danh sách ô (chỉ giữ ô trên đất liền) để web vẽ bản đồ

Mô hình được nạp 1 lần rồi giữ trong bộ nhớ (nhanh cho web).
"""

import numpy as np
import pandas as pd
import joblib

import config
from config import (
    LAT_MIN, LAT_MAX, LON_MIN, LON_MAX,
    GRID_STEP_DEG, MODEL_FILE,
)
from weather_api import load_weather_cache, fetch_weather_batched

# Nạp mô hình 1 lần (cache ở cấp module)
_BUNDLE = None


def _get_bundle():
    global _BUNDLE
    if _BUNDLE is None:
        _BUNDLE = joblib.load(MODEL_FILE)
    return _BUNDLE


def _build_grid(date, step=None):
    """Sinh lưới tâm ô đều phủ khung Việt Nam cho 1 ngày."""
    step = step or GRID_STEP_DEG
    lats = np.arange(LAT_MIN, LAT_MAX + 1e-9, step)
    lons = np.arange(LON_MIN, LON_MAX + 1e-9, step)
    glat, glon = np.meshgrid(lats, lons)
    month = int(date.split("-")[1])
    return pd.DataFrame({
        "latitude": np.round(glat.ravel(), 4),
        "longitude": np.round(glon.ravel(), 4),
        "acq_date": date,
        "month": month,
    })


def compute_grid(date, step=None, progress_cb=None):
    """
    Tính DataFrame lưới đầy đủ cho 1 ngày: mỗi tâm ô có tọa độ, thời tiết, độ cao
    và XÁC SUẤT nguy cơ (cột 'risk'). Ô nào KHÔNG lấy được thời tiết -> risk = NaN
    (để phần vẽ bản đồ nhiệt nội suy bỏ qua). Đây là hàm dùng chung cho cả bản đồ
    nhiệt (heatmap) lẫn API trả về ô.
    """
    bundle = _get_bundle()
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]
    medians = bundle["medians"]

    grid = _build_grid(date, step=step)

    # ================== NGUỒN THỜI TIẾT ==================
    # NGÀY TƯƠNG LAI (hôm nay → +15 ngày): dùng DỰ BÁO thời tiết Open-Meteo -> mô
    # hình DỰ BÁO nguy cơ cháy các ngày TỚI (biến hệ thống thành cảnh báo sớm).
    import datetime as _dt
    _today = _dt.date.today()
    try:
        _gdate = _dt.date.fromisoformat(str(date))
    except Exception:
        _gdate = _today
    is_forecast = _gdate >= (_today - _dt.timedelta(days=4))
    got_gee = False
    if is_forecast:
        try:
            from fetch_forecast import fetch_forecast_for_points
            g = fetch_forecast_for_points(grid[["latitude", "longitude", "acq_date"]])
            if g is not None and g["temp_max"].notna().any():
                for c in g.columns:
                    grid[c] = g[c].values
                got_gee = True
        except Exception:
            pass

    # NGÀY QUÁ KHỨ: lấy TẤT CẢ đặc trưng từ ERA5-Land qua Google Earth Engine —
    # MIỄN PHÍ, KHÔNG quota. (Open-Meteo archive vốn là ERA5 nên giá trị y hệt.)
    use_gee = getattr(config, "USE_GEE_WEATHER", True)
    if not got_gee and use_gee:
        try:
            from fetch_weather_gee import fetch_weather_gee_for_points
            g = fetch_weather_gee_for_points(
                grid[["latitude", "longitude", "acq_date"]], verbose=False
            )
            if g is not None and g["temp_max"].notna().any():
                for c in g.columns:
                    grid[c] = g[c].values
                got_gee = True
        except Exception:
            pass

    # Nếu KHÔNG dùng GEE (hoặc GEE lỗi) -> Open-Meteo, rồi phao GEE cho điểm thiếu.
    if not got_gee:
        cache = load_weather_cache()
        feats = fetch_weather_batched(
            grid[["latitude", "longitude", "acq_date"]], cache, progress_cb=progress_cb
        )
        for col in feats.columns:
            grid[col] = feats[col].values
        missing = grid["temp_max"].isna() if "temp_max" in grid else None
        if missing is not None and missing.any():
            try:
                from fetch_weather_gee import fetch_weather_gee_for_points
                g = fetch_weather_gee_for_points(
                    grid.loc[missing, ["latitude", "longitude", "acq_date"]], verbose=False
                )
                if g is not None:
                    for c in g.columns:
                        grid.loc[missing, c] = g[c].values
            except Exception:
                pass

    # NDVI (thảm thực vật) — luôn lấy từ GEE.
    if "ndvi" in feature_columns:
        try:
            from fetch_ndvi import init_ee, fetch_ndvi_for_points
            if init_ee():
                grid["ndvi"] = fetch_ndvi_for_points(
                    grid[["latitude", "longitude", "acq_date"]], verbose=False
                ).values
        except Exception:
            pass

    # Đảm bảo đủ mọi cột mô hình cần (cột thiếu -> NaN -> điền median lúc train)
    for c in feature_columns:
        if c not in grid.columns:
            grid[c] = np.nan

    X = grid[feature_columns].fillna(medians)
    grid["risk"] = model.predict_proba(X)[:, 1]
    # Ô không có dữ liệu thời tiết thật -> để risk = NaN (không bịa số)
    grid.loc[grid["temp_max"].isna(), "risk"] = np.nan
    return grid


def predict_grid_for_date(date, step=None, progress_cb=None):
    """
    Trả về dict {date, cells, step, stats, model_name}.
    Chỉ giữ ô trên ĐẤT LIỀN (độ cao >= 0) và có dữ liệu để bản đồ sạch.
    """
    bundle = _get_bundle()
    grid = compute_grid(date, step=step, progress_cb=progress_cb)

    # Giữ ô đất liền + có risk
    land = grid[(grid["elevation"].fillna(-9999) >= 0) & grid["risk"].notna()].copy()

    cells = []
    for r in land.itertuples(index=False):
        cells.append({
            "lat": float(r.latitude),
            "lon": float(r.longitude),
            "risk": round(float(r.risk), 3),
            "temp_max": None if pd.isna(r.temp_max) else float(r.temp_max),
            "humidity_mean": None if pd.isna(r.humidity_mean) else float(r.humidity_mean),
            "wind_max": None if pd.isna(r.wind_max) else float(r.wind_max),
            "precip_sum": None if pd.isna(r.precip_sum) else float(r.precip_sum),
        })

    stats = {
        "mean_risk": round(float(land["risk"].mean()), 3) if len(land) else 0.0,
        "max_risk": round(float(land["risk"].max()), 3) if len(land) else 0.0,
        "n_cells": int(len(land)),
        "n_high": int((land["risk"] >= 0.6).sum()),
    }

    return {
        "date": date,
        "cells": cells,
        "step": step or GRID_STEP_DEG,
        "stats": stats,
        "model_name": bundle.get("model_name", "?"),
    }
