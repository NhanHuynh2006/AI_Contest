# -*- coding: utf-8 -*-
"""
Bước 2 — generate_negatives.py: Sinh NHÃN ÂM (điểm "không cháy").

Ý tưởng (pseudo-absence sampling): dữ liệu NASA chỉ cho ta biết nơi ĐÃ cháy
(nhãn dương). Để dạy mô hình phân biệt, ta cần cả nơi KHÔNG cháy (nhãn âm).
Ta sinh ngẫu nhiên các điểm trong lãnh thổ Việt Nam, nhưng BẮT BUỘC mỗi điểm
phải cách MỌI điểm cháy ít nhất MIN_DISTANCE_KM để chắc chắn nó "không cháy".

Để kiểm tra khoảng cách nhanh (thay vì so từng cặp — O(n²) rất chậm), ta dùng
cây không gian scipy.spatial.cKDTree: quy đổi (lat, lon) sang tọa độ km gần đúng
rồi truy vấn điểm cháy gần nhất cho mỗi ứng viên.
"""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from config import (
    LAT_MIN, LAT_MAX, LON_MIN, LON_MAX,
    DATE_START, DATE_END,
    MIN_DISTANCE_KM, MAX_DISTANCE_KM, NEG_POS_RATIO,
    NEGATIVES_CSV, RANDOM_SEED,
)
from load_labels import load_labels

# 1 độ vĩ ≈ 111 km. Với kinh độ, phải nhân thêm cos(vĩ độ) vì các đường kinh
# tuyến co lại về phía cực. Ta lấy vĩ độ trung bình của VN để quy đổi gần đúng.
KM_PER_DEG_LAT = 111.0
_MEAN_LAT = (LAT_MIN + LAT_MAX) / 2.0
KM_PER_DEG_LON = 111.0 * np.cos(np.radians(_MEAN_LAT))


def _to_km_xy(lat, lon):
    """Quy đổi (lat, lon) -> (x_km, y_km) để tính khoảng cách Euclid xấp xỉ km."""
    x = lon * KM_PER_DEG_LON
    y = lat * KM_PER_DEG_LAT
    return np.column_stack([x, y])


def generate_negatives(positives=None, verbose=True):
    """
    Sinh điểm âm và lưu ra NEGATIVES_CSV. Trả về DataFrame điểm âm.
    Mỗi điểm âm có: latitude, longitude, acq_date, month, label=0.
    """
    if positives is None:
        positives = load_labels(verbose=False)

    rng = np.random.default_rng(RANDOM_SEED)

    n_neg_target = int(len(positives) * NEG_POS_RATIO)

    if verbose:
        print("=" * 60)
        print("BƯỚC 2: SINH NHÃN ÂM (điểm không cháy)")
        print("=" * 60)
        print(f"  Số điểm âm cần sinh: {n_neg_target:,} (tỉ lệ {NEG_POS_RATIO:g}:1)")
        print(f"  Vành đai hợp lệ quanh điểm cháy: {MIN_DISTANCE_KM}–{MAX_DISTANCE_KM} km")
        print(f"  (>= {MIN_DISTANCE_KM}km: chắc chắn 'không cháy'; "
              f"<= {MAX_DISTANCE_KM}km: ở lại đất liền, môi trường tương đồng)")

    # Xây cây KD trên các điểm cháy (đơn vị km)
    pos_xy = _to_km_xy(positives["latitude"].values, positives["longitude"].values)
    tree = cKDTree(pos_xy)

    # Dải ngày để gán ngẫu nhiên (cùng khoảng thời gian với dữ liệu cháy)
    date_range = pd.date_range(DATE_START, DATE_END, freq="D")

    accepted = []
    batch = max(n_neg_target, 2000)          # sinh theo lô để nhanh
    attempts = 0
    while len(accepted) < n_neg_target and attempts < 400:
        attempts += 1
        cand_lat = rng.uniform(LAT_MIN, LAT_MAX, batch)
        cand_lon = rng.uniform(LON_MIN, LON_MAX, batch)
        cand_xy = _to_km_xy(cand_lat, cand_lon)

        # Khoảng cách tới điểm cháy GẦN NHẤT cho từng ứng viên
        dist_km, _ = tree.query(cand_xy, k=1)
        # Vành đai: đủ xa để chắc chắn "không cháy", nhưng đủ gần để còn ở
        # trên đất liền (điểm cháy chỉ có trên đất) — tránh điểm âm rơi ra biển
        keep = (dist_km >= MIN_DISTANCE_KM) & (dist_km <= MAX_DISTANCE_KM)

        for la, lo in zip(cand_lat[keep], cand_lon[keep]):
            accepted.append((la, lo))
            if len(accepted) >= n_neg_target:
                break

    accepted = accepted[:n_neg_target]
    lats = np.array([p[0] for p in accepted])
    lons = np.array([p[1] for p in accepted])

    # Gán ngày ngẫu nhiên -> để bước 3 lấy đúng thời tiết ngày đó
    rand_dates = rng.choice(date_range, size=len(accepted))
    rand_dates = pd.to_datetime(rand_dates)

    neg = pd.DataFrame({
        "latitude": np.round(lats, 4),
        "longitude": np.round(lons, 4),
        "acq_date": rand_dates.strftime("%Y-%m-%d"),
        "month": rand_dates.month,
        "label": 0,
    })

    neg.to_csv(NEGATIVES_CSV, index=False)

    if verbose:
        print(f"  Đã sinh & chấp nhận: {len(neg):,} điểm âm")
        print(f"  Số lô đã thử: {attempts}")
        print(f"  Đã lưu: {NEGATIVES_CSV.name}")
        print()

    return neg


if __name__ == "__main__":
    generate_negatives()
