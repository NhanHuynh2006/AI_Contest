# -*- coding: utf-8 -*-
"""
weather_api.py — Hàm dùng chung để lấy thời tiết & độ cao từ Open-Meteo.

Được dùng bởi cả fetch_features.py (Bước 3) và make_risk_map.py (Bước 5).
API Open-Meteo MIỄN PHÍ, KHÔNG cần API key.

Điểm mấu chốt về TỐC ĐỘ:
  Gọi API cho từng điểm một sẽ rất chậm (hàng nghìn request). May mắn là
  Open-Meteo cho phép gửi NHIỀU TỌA ĐỘ trong 1 request (latitude=a,b,c...).
  Vì mỗi request chỉ nhận 1 khoảng ngày chung, ta GOM các điểm theo NGÀY rồi
  gửi theo lô (batch). Nhờ vậy ~4000 điểm chỉ còn ~200 request.

Ngoài ra module có:
  - CACHE ra file CSV (chạy lại không phải gọi API lại)
  - RETRY khi lỗi/timeout, nghỉ (delay) giữa các request để tránh rate-limit
"""

import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import (
    WEATHER_CACHE_CSV,
    API_DELAY_SEC,
    API_MAX_RETRY,
    API_TIMEOUT_SEC,
    PAST_DAYS,
)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
ELEVATION_URL = "https://api.open-meteo.com/v1/elevation"

# Số tọa độ tối đa gửi trong 1 request. Open-Meteo cho phép vài trăm tọa độ/1 lần
# -> gửi lô lớn để giảm SỐ LƯỢNG request (ít bị chặn hơn).
BATCH_SIZE = 200
# Số request chạy SONG SONG. Open-Meteo tính "trọng số" theo số tọa độ, nên nếu
# bắn quá nhiều cùng lúc sẽ bị chặn (429). 3 luồng là cân bằng nhanh/an toàn.
MAX_WORKERS = 3

# Các cột đặc trưng lưu trong cache (v2: có thêm mưa tích lũy & chuỗi ngày khô)
_CACHE_FIELDS = [
    "temp_max", "humidity_mean", "wind_max", "precip_sum",
    "precip_7d", "precip_14d", "dry_days_14d", "elevation",
]

# Chuỗi biến daily gửi lên API (lấy cả cửa sổ PAST_DAYS+1 ngày)
_DAILY_VARS = ("temperature_2m_max,relative_humidity_2m_mean,"
               "wind_speed_10m_max,precipitation_sum")


def _window_start(date):
    """Ngày bắt đầu cửa sổ thời tiết: PAST_DAYS ngày trước ngày quan sát."""
    d = datetime.strptime(date, "%Y-%m-%d") - timedelta(days=PAST_DAYS)
    return d.strftime("%Y-%m-%d")


# ----------------------------------------------------------------------------
# CACHE
# ----------------------------------------------------------------------------
def _cacheckey(lat, lon, date):
    """Khóa cache: làm tròn tọa độ 3 chữ số (~100m) + ngày."""
    return f"{round(float(lat), 3)}_{round(float(lon), 3)}_{date}"


def load_weather_cache():
    """Đọc cache thời tiết từ file (nếu có). Trả về dict {key: {...}}."""
    if WEATHER_CACHE_CSV.exists():
        df = pd.read_csv(WEATHER_CACHE_CSV)
        cache = {}
        for _, r in df.iterrows():
            cache[str(r["key"])] = {f: r[f] for f in _CACHE_FIELDS}
        return cache
    return {}


def save_weather_cache(cache):
    """Ghi toàn bộ cache ra file CSV."""
    rows = [{"key": k, **v} for k, v in cache.items()]
    pd.DataFrame(rows).to_csv(WEATHER_CACHE_CSV, index=False)


# ----------------------------------------------------------------------------
# GỌI API (có retry)
# ----------------------------------------------------------------------------
# Chế độ "thất bại nhanh" cho WEB: khi bị rate-limit (429) thì bỏ cuộc NGAY thay vì
# chờ retry tới vài phút (offline pipeline thì tắt cờ này để kiên nhẫn chờ quota).
_FAST_FAIL = False
_RATE_LIMITED = False        # đã phát hiện bị chặn trong lần fetch hiện tại?


def set_fast_fail(v=True):
    """Bật/tắt chế độ thất bại nhanh (web bật, pipeline offline tắt)."""
    global _FAST_FAIL
    _FAST_FAIL = v


def rate_limited():
    """True nếu lần fetch vừa rồi bị Open-Meteo chặn (để web báo lỗi rõ)."""
    return _RATE_LIMITED


def _request_json(url, params):
    """Gọi API có retry + backoff. Trả về JSON hoặc None nếu thất bại hẳn."""
    global _RATE_LIMITED
    if _FAST_FAIL and _RATE_LIMITED:
        return None                     # đã biết bị chặn -> khỏi gọi thêm (nhanh)
    for attempt in range(1, API_MAX_RETRY + 1):
        try:
            resp = requests.get(url, params=params, timeout=API_TIMEOUT_SEC)
            if resp.status_code == 200:
                return resp.json()
            # 429 = bị rate-limit (hết lượt gọi theo phút/ngày).
            if resp.status_code == 429:
                _RATE_LIMITED = True
                if _FAST_FAIL:
                    return None          # web: bỏ cuộc ngay, báo lỗi thân thiện
                time.sleep(min(60, 12 * attempt))   # pipeline: kiên nhẫn chờ
                continue
        except requests.RequestException:
            if _FAST_FAIL:
                return None
            time.sleep(2.0 * attempt)
    return None


def _parse_daily(entry):
    """
    Trích 1 phần tử JSON (1 tọa độ, chuỗi 14 ngày) thành dict đặc trưng.

    Ngày QUAN SÁT là phần tử CUỐI của mỗi mảng daily. Các ngày trước đó dùng để
    tính đặc trưng tích lũy: tổng mưa 7/14 ngày và số ngày khô — đây là các chỉ số
    "độ khô nhiên liệu" mà những mô hình cảnh báo cháy chuyên nghiệp đều dùng.
    """
    result = {f: None for f in _CACHE_FIELDS}
    if not entry:
        return result
    daily = entry.get("daily") or {}

    def _last(name):
        vals = daily.get(name)
        return vals[-1] if vals else None

    result["temp_max"] = _last("temperature_2m_max")
    result["humidity_mean"] = _last("relative_humidity_2m_mean")
    result["wind_max"] = _last("wind_speed_10m_max")

    pr = daily.get("precipitation_sum") or []
    pr_ok = [p for p in pr if p is not None]
    if pr:
        result["precip_sum"] = pr[-1]
        last7 = [p for p in pr[-7:] if p is not None]
        result["precip_7d"] = round(sum(last7), 2) if last7 else None
        result["precip_14d"] = round(sum(pr_ok), 2) if pr_ok else None
        result["dry_days_14d"] = sum(1 for p in pr_ok if p < 1.0) if pr_ok else None

    result["elevation"] = entry.get("elevation")
    return result


# ----------------------------------------------------------------------------
# LẤY THEO LÔ (gom theo ngày) — cách nhanh, dùng chính
# ----------------------------------------------------------------------------
def fetch_weather_batched(points, cache, progress_cb=None):
    """
    Lấy đặc trưng cho nhiều điểm cùng lúc.

    points: DataFrame có cột latitude, longitude, acq_date.
    cache : dict cache (sẽ được cập nhật tại chỗ).
    progress_cb: hàm gọi lại nhận số điểm vừa xử lý (để cập nhật tqdm).

    Trả về DataFrame đặc trưng THEO ĐÚNG THỨ TỰ input (cùng index).
    """
    global _RATE_LIMITED
    _RATE_LIMITED = False              # đặt lại cờ rate-limit cho mỗi lần fetch
    df = points.reset_index(drop=True).copy()
    df["ckey"] = [
        _cacheckey(la, lo, d)
        for la, lo, d in zip(df["latitude"], df["longitude"], df["acq_date"])
    ]

    # Chỉ cần gọi API cho những điểm CHƯA có trong cache
    missing = df[~df["ckey"].isin(cache.keys())]

    # Báo tiến độ ngay cho phần đã có sẵn trong cache (không phải gọi API)
    if progress_cb:
        progress_cb(len(df) - len(missing))

    # Gom điểm còn thiếu theo NGÀY, mỗi ngày cắt thành các lô BATCH_SIZE tọa độ.
    # Mỗi lô là 1 "công việc" (task) sẽ được gọi API song song bằng nhiều luồng.
    tasks = []
    for date, grp in missing.groupby("acq_date"):
        grp = grp.reset_index(drop=True)
        for start in range(0, len(grp), BATCH_SIZE):
            tasks.append((date, grp.iloc[start:start + BATCH_SIZE]))

    def _do_task(task):
        """Gọi API cho 1 lô, trả về (danh sách khóa, danh sách đặc trưng)."""
        date, chunk = task
        params = {
            "latitude": ",".join(str(x) for x in chunk["latitude"]),
            "longitude": ",".join(str(x) for x in chunk["longitude"]),
            "start_date": _window_start(date),   # lấy cả 13 ngày trước đó
            "end_date": date,
            "daily": _DAILY_VARS,
            "timezone": "auto",
        }
        data = _request_json(ARCHIVE_URL, params)
        if isinstance(data, dict):        # 1 tọa độ -> dict; nhiều -> list
            data = [data]
        if data is None:                  # lỗi hẳn -> để None, xử lý ở bước train
            data = [None] * len(chunk)
        keys = list(chunk["ckey"])
        return keys, [_parse_daily(e) for e in data]

    # Chạy các task song song. Việc GHI vào cache chỉ làm ở luồng chính -> an toàn.
    # CHỈ lưu cache những điểm LẤY ĐƯỢC (temp_max khác None); điểm lỗi để trống
    # để lần chạy sau tự thử lại (tránh "đầu độc" cache bằng giá trị rỗng).
    since_save = 0
    if tasks:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = [pool.submit(_do_task, t) for t in tasks]
            for fut in as_completed(futures):
                keys, results = fut.result()
                for k, r in zip(keys, results):
                    if r.get("temp_max") is not None:   # chỉ cache khi có dữ liệu thật
                        cache[k] = r
                since_save += len(keys)
                if progress_cb:
                    progress_cb(len(keys))
                if since_save >= 500:     # định kỳ lưu cache
                    save_weather_cache(cache)
                    since_save = 0

    save_weather_cache(cache)

    # Ghép kết quả theo đúng thứ tự input. Điểm nào không lấy được (không có trong
    # cache) -> trả dict rỗng (toàn None) để bước sau xử lý giá trị thiếu.
    empty = {f: None for f in _CACHE_FIELDS}
    feats = pd.DataFrame([cache.get(k, empty) for k in df["ckey"]])
    feats.index = df.index
    return feats


# ----------------------------------------------------------------------------
# LẤY 1 ĐIỂM (đơn giản, để tham khảo / dự phòng)
# ----------------------------------------------------------------------------
def fetch_weather_point(lat, lon, date, cache):
    """Lấy đặc trưng cho đúng 1 điểm. Ưu tiên cache; nếu thiếu thì gọi API."""
    key = _cacheckey(lat, lon, date)
    if key in cache:
        return cache[key]
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": _window_start(date), "end_date": date,
        "daily": _DAILY_VARS,
        "timezone": "auto",
    }
    data = _request_json(ARCHIVE_URL, params)
    result = _parse_daily(data if isinstance(data, dict) else None)
    cache[key] = result
    time.sleep(API_DELAY_SEC)
    return result
