# -*- coding: utf-8 -*-
"""
fetch_rain_gee.py — Lấy đặc trưng "mưa tích lũy" từ Google Earth Engine (ERA5-Land).

Vì API thời tiết miễn phí Open-Meteo giới hạn số lần gọi/ngày, ta lấy lượng mưa
14 ngày trước mỗi điểm từ ERA5-Land (ECMWF) qua GEE — nguồn tái phân tích toàn cầu,
KHÔNG giới hạn quota. Trả về 3 đặc trưng giống pipeline chính:
    precip_7d      tổng mưa 7 ngày gần nhất (mm)
    precip_14d     tổng mưa 14 ngày gần nhất (mm)
    dry_days_14d   số ngày khô (<1mm) trong 14 ngày

Dùng cho GRID bản đồ web (mỗi khung 1 ngày -> 1 cửa sổ, rất ít lần gọi GEE).
"""

import pandas as pd
from datetime import datetime, timedelta
from config import PAST_DAYS
from fetch_ndvi import init_ee, GEE_PROJECT   # tái dùng khởi tạo GEE

ERA5 = "ECMWF/ERA5_LAND/DAILY_AGGR"
CHUNK = 800


def _rain_images(date):
    """Ảnh tổng mưa 14 ngày, 7 ngày, và số ngày khô — cho 1 ngày quan sát."""
    import ee
    end = datetime.strptime(date, "%Y-%m-%d")
    start14 = (end - timedelta(days=PAST_DAYS)).strftime("%Y-%m-%d")
    start7 = (end - timedelta(days=6)).strftime("%Y-%m-%d")
    endp = (end + timedelta(days=1)).strftime("%Y-%m-%d")   # end exclusive

    col14 = (ee.ImageCollection(ERA5).filterDate(start14, endp)
             .select("total_precipitation_sum"))
    col7 = (ee.ImageCollection(ERA5).filterDate(start7, endp)
            .select("total_precipitation_sum"))
    p14 = col14.sum().multiply(1000).rename("precip_14d")     # m -> mm
    p7 = col7.sum().multiply(1000).rename("precip_7d")
    # ngày khô: mỗi ảnh ngày, mưa(mm) < 1 -> 1, cộng lại
    dry = (col14.map(lambda im: im.multiply(1000).lt(1.0))
           .sum().rename("dry_days_14d"))
    return p14.addBands(p7).addBands(dry)


def fetch_rain_for_points(points, verbose=True):
    """
    Trả về DataFrame (precip_7d, precip_14d, dry_days_14d) theo thứ tự input.
    Gom điểm theo NGÀY quan sát -> mỗi ngày 1 lần dựng ảnh (hiệu quả cho grid).
    """
    import ee
    df = points.reset_index(drop=True).copy()
    out = pd.DataFrame(index=df.index,
                       columns=["precip_7d", "precip_14d", "dry_days_14d"], dtype=float)

    for date, grp in df.groupby(df["acq_date"].astype(str)):
        img = _rain_images(date)
        idxs = grp.index.tolist()
        for s in range(0, len(idxs), CHUNK):
            sub = grp.iloc[s:s + CHUNK]
            feats = [ee.Feature(ee.Geometry.Point(float(r.longitude), float(r.latitude)),
                                {"i": int(i)})
                     for i, r in zip(sub.index, sub.itertuples(index=False))]
            fc = ee.FeatureCollection(feats)
            try:
                res = img.reduceRegions(collection=fc, reducer=ee.Reducer.first(),
                                        scale=1000).getInfo()
                for f in res["features"]:
                    p = f["properties"]
                    i = p["i"]
                    out.at[i, "precip_7d"] = p.get("precip_7d")
                    out.at[i, "precip_14d"] = p.get("precip_14d")
                    out.at[i, "dry_days_14d"] = p.get("dry_days_14d")
            except Exception as e:
                if verbose:
                    print(f"    (lỗi GEE ngày {date}: {e})")
        if verbose:
            print(f"  mưa GEE: xong ngày {date} ({len(idxs)} điểm)", flush=True)
    return out.round(2)
