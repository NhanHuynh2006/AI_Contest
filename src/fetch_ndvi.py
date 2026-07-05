# -*- coding: utf-8 -*-
"""
fetch_ndvi.py — Lấy NDVI (chỉ số thảm thực vật) từ Google Earth Engine.

NDVI đo "độ xanh" của thảm thực vật (0=trơ trọi, ~0.9=rừng rậm). Nghiên cứu tại
Việt Nam xếp NDVI là biến dự báo cháy MẠNH NHẤT: nơi nhiều sinh khối khô mới có
nhiên liệu để cháy. Nguồn: ảnh vệ tinh MODIS MOD13Q1 (250m, chu kỳ 16 ngày) —
cùng họ vệ tinh với dữ liệu điểm cháy FIRMS nên rất nhất quán.

Đây là MODULE TÙY CHỌN: nếu chưa xác thực Google Earth Engine, pipeline vẫn chạy
bình thường (không có cột ndvi). Xác thực 1 lần duy nhất bằng lệnh:
    earthengine authenticate

Cách dùng:
    python src/fetch_ndvi.py     # thêm cột 'ndvi' vào dataset_with_features.csv
"""

import pandas as pd
from config import DATASET_CSV, CACHE_DIR

# Project Google Cloud đã đăng ký Earth Engine của đội
GEE_PROJECT = "aicontest-501322"
NDVI_CACHE_CSV = CACHE_DIR / "ndvi_cache.csv"

# Số điểm tối đa gửi trong 1 lần gọi Earth Engine
CHUNK = 800

_EE_READY = None      # cache trạng thái khởi tạo (None = chưa thử)


def init_ee(verbose=False):
    """Khởi tạo Earth Engine. Trả về True/False. An toàn để gọi nhiều lần."""
    global _EE_READY
    if _EE_READY is not None:
        return _EE_READY
    try:
        import ee
        ee.Initialize(project=GEE_PROJECT)
        _EE_READY = True
    except Exception as e:
        if verbose:
            print(f"  (GEE chưa sẵn sàng: {type(e).__name__}. "
                  f"Chạy 'earthengine authenticate' một lần rồi thử lại.)")
        _EE_READY = False
    return _EE_READY


# ----------------------------------------------------------------------------
# CACHE (giống cơ chế cache thời tiết: chạy lại không phải gọi GEE lại)
# ----------------------------------------------------------------------------
def _key(lat, lon, ym):
    """Khóa cache: tọa độ làm tròn ~100m + THÁNG (NDVI đổi chậm, theo tháng đủ)."""
    return f"{round(float(lat), 3)}_{round(float(lon), 3)}_{ym}"


def _load_cache():
    if NDVI_CACHE_CSV.exists():
        df = pd.read_csv(NDVI_CACHE_CSV)
        return dict(zip(df["key"].astype(str), df["ndvi"]))
    return {}


def _save_cache(cache):
    pd.DataFrame({"key": list(cache.keys()), "ndvi": list(cache.values())}
                 ).to_csv(NDVI_CACHE_CSV, index=False)


# ----------------------------------------------------------------------------
# LẤY NDVI THEO LÔ (gom theo tháng -> 1 ảnh composite / tháng)
# ----------------------------------------------------------------------------
def fetch_ndvi_for_points(points, verbose=True):
    """
    Lấy NDVI cho từng điểm (lat, lon, acq_date).

    Trả về pandas Series NDVI theo đúng thứ tự input (NaN nếu không lấy được).
    Cách làm: gom điểm theo THÁNG; với mỗi tháng dựng ảnh NDVI trung vị từ các
    composite 16 ngày trong ~2 tháng gần đó, rồi đọc giá trị tại loạt điểm
    (reduceRegions) — mỗi tháng chỉ tốn 1-vài lần gọi API nên rất nhanh.
    """
    import ee

    df = points.reset_index(drop=True).copy()
    df["ym"] = df["acq_date"].astype(str).str[:7]
    df["k"] = [_key(la, lo, ym) for la, lo, ym in
               zip(df["latitude"], df["longitude"], df["ym"])]

    cache = _load_cache()
    missing = df[~df["k"].isin(cache.keys())]

    groups = list(missing.groupby("ym"))
    if verbose and groups:
        print(f"  NDVI: cần lấy {len(missing):,} điểm / {len(groups)} tháng "
              f"(đã có {len(df) - len(missing):,} trong cache)")

    for gi, (ym, grp) in enumerate(groups, 1):
        # Cửa sổ ảnh: từ đầu tháng TRƯỚC đến hết tháng hiện tại (đảm bảo luôn có
        # ít nhất 2-4 composite 16 ngày để lấy trung vị, kể cả khi thiếu ảnh)
        p = pd.Period(ym)
        start = (p - 1).to_timestamp().strftime("%Y-%m-%d")
        end = (p + 1).to_timestamp().strftime("%Y-%m-%d")
        img = (ee.ImageCollection("MODIS/061/MOD13Q1")
               .filterDate(start, end).select("NDVI").median()
               .multiply(0.0001))          # về thang chuẩn -1..1

        for s in range(0, len(grp), CHUNK):
            chunk = grp.iloc[s:s + CHUNK]
            feats = [ee.Feature(ee.Geometry.Point(float(r.longitude), float(r.latitude)),
                                {"k": r.k})
                     for r in chunk.itertuples(index=False)]
            fc = ee.FeatureCollection(feats)
            try:
                res = img.reduceRegions(collection=fc, reducer=ee.Reducer.first(),
                                        scale=250).getInfo()
                for f in res["features"]:
                    cache[f["properties"]["k"]] = f["properties"].get("first")
            except Exception as e:
                if verbose:
                    print(f"    (lỗi GEE tháng {ym}: {e} — bỏ qua, sẽ thử lại lần sau)")
        if verbose and gi % 10 == 0:
            print(f"    ... xong {gi}/{len(groups)} tháng")
            _save_cache(cache)

    _save_cache(cache)
    return df["k"].map(lambda k: cache.get(k)).astype(float)


def add_ndvi_to_dataset(verbose=True):
    """Đọc dataset_with_features.csv, thêm/cập nhật cột 'ndvi', lưu lại."""
    if verbose:
        print("=" * 60)
        print("BƯỚC 3b (TÙY CHỌN): LẤY NDVI TỪ GOOGLE EARTH ENGINE")
        print("=" * 60)
    if not init_ee(verbose=True):
        print("  -> Bỏ qua NDVI (chưa xác thực GEE). Pipeline vẫn chạy bình thường.")
        return False

    df = pd.read_csv(DATASET_CSV)
    ndvi = fetch_ndvi_for_points(df[["latitude", "longitude", "acq_date"]],
                                 verbose=verbose)
    df["ndvi"] = ndvi.round(4).values
    df.to_csv(DATASET_CSV, index=False)

    if verbose:
        ok = df["ndvi"].notna().mean() * 100
        print(f"  Đã thêm cột ndvi: {ok:.1f}% điểm có giá trị")
        print(f"  NDVI trung bình — điểm cháy: {df[df.label==1]['ndvi'].mean():.3f} | "
              f"điểm không cháy: {df[df.label==0]['ndvi'].mean():.3f}")
        print(f"  Đã lưu: {DATASET_CSV.name}")
    return True


if __name__ == "__main__":
    add_ndvi_to_dataset()
