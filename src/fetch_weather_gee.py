# -*- coding: utf-8 -*-
"""
fetch_weather_gee.py — Lấy TOÀN BỘ đặc trưng thời tiết từ ERA5-Land qua Google
Earth Engine, để KHÔNG phụ thuộc hạn mức (quota) của Open-Meteo.

Vì sao dùng được: Open-Meteo archive vốn lấy từ tái phân tích ERA5, nên lấy trực
tiếp ERA5-Land (ECMWF) cho ra giá trị gần như y hệt — mà GEE không giới hạn ngặt
theo ngày như Open-Meteo. Đây là "phao cứu sinh" cho ngày thi: nếu Open-Meteo hết
lượt, web tự chuyển sang nguồn này và vẫn chạy bình thường.

Trả về đủ 8 cột: temp_max, humidity_mean, wind_max, precip_sum, precip_7d,
precip_14d, dry_days_14d, elevation.
"""

import pandas as pd
from datetime import datetime, timedelta
from config import PAST_DAYS
from fetch_ndvi import init_ee          # tái dùng khởi tạo GEE

ERA5 = "ECMWF/ERA5_LAND/DAILY_AGGR"
CHUNK = 800
_COLS = ["temp_max", "humidity_mean", "wind_max", "precip_sum",
         "precip_7d", "precip_14d", "dry_days_14d", "elevation"]


def _weather_image(date):
    """ee.Image gồm 8 band đặc trưng cho 1 ngày quan sát (đơn vị đã đổi chuẩn)."""
    import ee
    end = datetime.strptime(date, "%Y-%m-%d")
    day0 = end.strftime("%Y-%m-%d")
    endp = (end + timedelta(days=1)).strftime("%Y-%m-%d")     # end exclusive
    start14 = (end - timedelta(days=PAST_DAYS)).strftime("%Y-%m-%d")
    start7 = (end - timedelta(days=6)).strftime("%Y-%m-%d")

    day = ee.ImageCollection(ERA5).filterDate(day0, endp).first()
    tmax = day.select("temperature_2m_max").subtract(273.15).rename("temp_max")
    t = day.select("temperature_2m").subtract(273.15)          # °C
    td = day.select("dewpoint_temperature_2m").subtract(273.15)

    # Độ ẩm tương đối từ nhiệt độ & điểm sương (công thức Magnus)
    def esat(x):
        return x.multiply(17.625).divide(x.add(243.04)).exp()
    rh = esat(td).divide(esat(t)).multiply(100).clamp(0, 100).rename("humidity_mean")

    u = day.select("u_component_of_wind_10m_max")
    v = day.select("v_component_of_wind_10m_max")
    wind = u.hypot(v).multiply(3.6).rename("wind_max")         # m/s -> km/h
    pr = day.select("total_precipitation_sum").multiply(1000).rename("precip_sum")  # m->mm

    # Đặc trưng tích lũy (mưa 7/14 ngày, số ngày khô)
    col14 = ee.ImageCollection(ERA5).filterDate(start14, endp).select("total_precipitation_sum")
    col7 = ee.ImageCollection(ERA5).filterDate(start7, endp).select("total_precipitation_sum")
    p14 = col14.sum().multiply(1000).rename("precip_14d")
    p7 = col7.sum().multiply(1000).rename("precip_7d")
    dry = col14.map(lambda im: im.multiply(1000).lt(1.0)).sum().rename("dry_days_14d")

    elev = ee.Image("USGS/SRTMGL1_003").rename("elevation")

    return tmax.addBands([rh, wind, pr, p7, p14, dry, elev])


def fetch_weather_gee_for_points(points, verbose=True):
    """
    Trả DataFrame 8 cột đặc trưng cho từng điểm (lat, lon, acq_date), đúng thứ tự
    input. Gom theo NGÀY -> mỗi ngày 1 lần dựng ảnh (rất ít lần gọi GEE cho grid).
    """
    if not init_ee():
        return None
    import ee
    df = points.reset_index(drop=True).copy()
    out = pd.DataFrame(index=df.index, columns=_COLS, dtype=float)

    for date, grp in df.groupby(df["acq_date"].astype(str)):
        img = _weather_image(date)
        idxs = grp.index.tolist()
        for s in range(0, len(idxs), CHUNK):
            sub = grp.iloc[s:s + CHUNK]
            feats = [ee.Feature(ee.Geometry.Point(float(r.longitude), float(r.latitude)),
                                {"i": int(i)})
                     for i, r in zip(sub.index, sub.itertuples(index=False))]
            fc = ee.FeatureCollection(feats)
            try:
                res = img.reduceRegions(collection=fc, reducer=ee.Reducer.first(),
                                        scale=10000).getInfo()
                for f in res["features"]:
                    p = f["properties"]
                    i = p["i"]
                    for c in _COLS:
                        if p.get(c) is not None:
                            out.at[i, c] = p[c]
            except Exception as e:
                if verbose:
                    print(f"    (lỗi GEE thời tiết ngày {date}: {e})")
        if verbose:
            print(f"  thời tiết GEE: xong ngày {date} ({len(idxs)} điểm)", flush=True)
    return out.round(2)
