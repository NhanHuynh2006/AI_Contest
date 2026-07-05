# -*- coding: utf-8 -*-
"""
Bước 3 — fetch_features.py: Lấy ĐẶC TRƯNG MÔI TRƯỜNG cho mọi điểm.

Gộp điểm dương (đã cháy) + điểm âm (không cháy), rồi với mỗi điểm lấy thời tiết
tại đúng (tọa độ, ngày) từ Open-Meteo, cộng thêm độ cao địa hình. Kết quả là
bảng học máy hoàn chỉnh: các cột đặc trưng + cột label.

Vì gọi API cho hàng nghìn điểm rất lâu và dễ bị rate-limit, module này:
  - lấy MẪU CON (SAMPLE_PER_CLASS trong config) để demo nhanh — đặt None để chạy hết
  - CACHE kết quả ra file (chạy lại không phải gọi lại API)
  - RETRY khi lỗi, có thanh tiến độ tqdm
  - điểm nào API lỗi thì để giá trị thiếu, bước train sẽ xử lý
"""

import pandas as pd
from tqdm import tqdm

from config import (
    LABELS_CSV, NEGATIVES_CSV, DATASET_CSV,
    SAMPLE_PER_CLASS, RANDOM_SEED, FEATURE_COLUMNS,
)
from weather_api import load_weather_cache, fetch_weather_batched


def _prepare_points():
    """
    Đọc điểm dương + âm, lấy mẫu con, rồi gộp lại.

    Kỹ thuật "time-matched pseudo-absence": điểm ÂM được gán NGÀY rút từ đúng
    phân phối ngày của điểm DƯƠNG đã lấy mẫu. Nhờ vậy:
      1. Hai lớp có cùng phân phối thời gian -> mô hình không thể "ăn gian" bằng
         cách đoán theo mùa, mà buộc phải học từ THỜI TIẾT thực sự.
      2. Số ngày khác nhau ít đi -> gom lô theo ngày hiệu quả hơn, giảm nửa số
         request API.
    """
    import numpy as np
    pos = pd.read_csv(LABELS_CSV)[["latitude", "longitude", "acq_date", "month", "label"]]
    neg = pd.read_csv(NEGATIVES_CSV)[["latitude", "longitude", "acq_date", "month", "label"]]

    if SAMPLE_PER_CLASS is not None:
        n = min(SAMPLE_PER_CLASS, len(pos), len(neg))
        pos = pos.sample(n=n, random_state=RANDOM_SEED)
        neg = neg.sample(n=n, random_state=RANDOM_SEED)

    # Gán lại ngày cho điểm âm: rút (có hoàn lại) từ ngày của điểm dương đã lấy
    rng = np.random.default_rng(RANDOM_SEED)
    matched_dates = rng.choice(pos["acq_date"].values, size=len(neg))
    neg = neg.copy()
    neg["acq_date"] = matched_dates
    neg["month"] = pd.to_datetime(neg["acq_date"]).dt.month

    data = pd.concat([pos, neg], ignore_index=True)
    # Trộn đều để dương/âm không nằm tách khối (đẹp hơn khi theo dõi)
    data = data.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    return data


def fetch_features(verbose=True):
    """Lấy đặc trưng cho toàn bộ điểm, lưu DATASET_CSV, trả về DataFrame."""
    data = _prepare_points()

    if verbose:
        print("=" * 60)
        print("BƯỚC 3: LẤY ĐẶC TRƯNG MÔI TRƯỜNG (Open-Meteo)")
        print("=" * 60)
        n_pos = int((data["label"] == 1).sum())
        n_neg = int((data["label"] == 0).sum())
        print(f"  Số điểm sẽ lấy: {len(data):,}  (dương={n_pos:,}, âm={n_neg:,})")
        if SAMPLE_PER_CLASS is not None:
            print(f"  (Đang chạy chế độ DEMO: {SAMPLE_PER_CLASS} điểm/lớp. "
                  f"Đặt SAMPLE_PER_CLASS=None trong config để chạy toàn bộ.)")
        print("  Có CACHE + RETRY: lần chạy sau sẽ nhanh hơn nhiều.")
        print()

    cache = load_weather_cache()
    if verbose and cache:
        print(f"  Đã nạp {len(cache):,} điểm từ cache.")

    # Lấy thời tiết theo LÔ (gom theo ngày) — nhanh hơn nhiều so với từng điểm
    if verbose:
        bar = tqdm(total=len(data), desc="  Lấy thời tiết", ncols=80)
        progress_cb = bar.update
    else:
        progress_cb = None

    feats = fetch_weather_batched(data[["latitude", "longitude", "acq_date"]],
                                  cache, progress_cb=progress_cb)
    if verbose:
        bar.close()

    # Ghép đặc trưng vào bảng gốc
    df = data.copy().reset_index(drop=True)
    for col in feats.columns:
        df[col] = feats[col].values

    # Sắp cột gọn gàng: định danh + đặc trưng + label
    id_cols = ["latitude", "longitude", "acq_date"]
    ordered = id_cols + FEATURE_COLUMNS + ["label"]
    df = df[[c for c in ordered if c in df.columns]]
    df.to_csv(DATASET_CSV, index=False)

    if verbose:
        n_missing = int(df[FEATURE_COLUMNS].isna().any(axis=1).sum())
        print()
        print(f"  Hoàn tất. Bảng đặc trưng: {df.shape[0]:,} dòng × {df.shape[1]} cột")
        print(f"  Số dòng có ít nhất 1 giá trị thiếu (API lỗi): {n_missing:,}")
        print(f"  Đã lưu: {DATASET_CSV.name}")
        print()

    return df


if __name__ == "__main__":
    fetch_features()
