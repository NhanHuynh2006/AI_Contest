# -*- coding: utf-8 -*-
"""
fetch_forecast.py — Lấy DỰ BÁO thời tiết (tương lai) từ Open-Meteo Forecast API,
để mô hình dự báo NGUY CƠ CHÁY cho các ngày TỚI (hôm nay → +15 ngày).

Đây là điểm mấu chốt biến hệ thống từ "xem lại quá khứ" thành "CẢNH BÁO SỚM":
mô hình học quan hệ môi trường→cháy từ dữ liệu lịch sử, rồi áp lên DỰ BÁO thời
tiết để ước lượng nơi nào sắp có nguy cơ cao trong những ngày tới.

Trả về đủ 8 cột như nguồn lịch sử: temp_max, humidity_mean, wind_max, precip_sum,
precip_7d, precip_14d, dry_days_14d, elevation.
"""

import time
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
BATCH_SIZE = 200
MAX_WORKERS = 3
_COLS = ["temp_max", "humidity_mean", "wind_max", "precip_sum",
         "precip_7d", "precip_14d", "dry_days_14d", "elevation"]


def _parse_point(entry, target_date):
    """Trích đặc trưng cho NGÀY mục tiêu từ mảng dự báo nhiều ngày của 1 tọa độ."""
    out = {c: None for c in _COLS}
    if not entry:
        return out
    daily = entry.get("daily") or {}
    times = daily.get("time") or []
    if target_date not in times:
        return out
    i = times.index(target_date)

    def at(name):
        arr = daily.get(name) or []
        return arr[i] if i < len(arr) else None

    out["temp_max"] = at("temperature_2m_max")
    out["humidity_mean"] = at("relative_humidity_2m_mean")
    out["wind_max"] = at("wind_speed_10m_max")
    pr = daily.get("precipitation_sum") or []
    if pr and i < len(pr):
        out["precip_sum"] = pr[i]
        w14 = [p for p in pr[max(0, i - 13):i + 1] if p is not None]
        w7 = [p for p in pr[max(0, i - 6):i + 1] if p is not None]
        out["precip_14d"] = round(sum(w14), 2) if w14 else None
        out["precip_7d"] = round(sum(w7), 2) if w7 else None
        out["dry_days_14d"] = sum(1 for p in w14 if p < 1.0) if w14 else None
    out["elevation"] = entry.get("elevation")
    return out


def fetch_forecast_for_points(points, verbose=False):
    """
    Lấy dự báo cho từng điểm (lat, lon, acq_date) — acq_date phải là ngày TƯƠNG LAI
    trong tầm ~15 ngày. Trả DataFrame 8 cột đúng thứ tự input (NaN nếu không có).
    """
    df = points.reset_index(drop=True).copy()
    target = str(df["acq_date"].iloc[0])            # 1 ngày chung cho cả lưới
    out = pd.DataFrame(index=df.index, columns=_COLS, dtype=float)

    tasks = []
    for s in range(0, len(df), BATCH_SIZE):
        tasks.append(df.iloc[s:s + BATCH_SIZE])

    def _do(chunk):
        params = {
            "latitude": ",".join(str(x) for x in chunk["latitude"]),
            "longitude": ",".join(str(x) for x in chunk["longitude"]),
            "daily": "temperature_2m_max,relative_humidity_2m_mean,"
                     "wind_speed_10m_max,precipitation_sum",
            "past_days": 15, "forecast_days": 16, "timezone": "auto",
        }
        for attempt in range(3):
            try:
                r = requests.get(FORECAST_URL, params=params, timeout=40)
                if r.status_code == 200:
                    data = r.json()
                    return chunk.index.tolist(), (data if isinstance(data, list) else [data])
                if r.status_code == 429:
                    time.sleep(3 * (attempt + 1)); continue
            except requests.RequestException:
                time.sleep(2)
        return chunk.index.tolist(), None

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = [ex.submit(_do, c) for c in tasks]
        for fu in as_completed(futs):
            idxs, entries = fu.result()
            if not entries:
                continue
            for k, i in enumerate(idxs):
                vals = _parse_point(entries[k] if k < len(entries) else None, target)
                for c in _COLS:
                    out.at[i, c] = vals[c]
    return out.round(2)
