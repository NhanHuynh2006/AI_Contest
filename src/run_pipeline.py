# -*- coding: utf-8 -*-
"""
run_pipeline.py — Chạy TOÀN BỘ pipeline dự báo nguy cơ cháy theo thứ tự.

Cách dùng:
    python src/run_pipeline.py            # chạy tuần tự cả 5 bước
    python src/run_pipeline.py --step 2   # chỉ chạy 1 bước (1..5)
    python src/run_pipeline.py --from 3   # chạy từ bước 3 tới hết

5 bước:
    1. load_labels        — nạp & thống kê nhãn dương (điểm đã cháy)
    2. generate_negatives — sinh nhãn âm (điểm không cháy)
    3. fetch_features     — lấy đặc trưng môi trường (Open-Meteo)
    4. train_model        — huấn luyện & đánh giá (RF + XGBoost)
    5. make_risk_map      — tạo bản đồ nguy cơ (HTML)
"""

import argparse
import time

from load_labels import load_labels
from generate_negatives import generate_negatives
from fetch_features import fetch_features
from train_model import train_model
from make_risk_map import make_risk_map


def run_all(start=1, only=None):
    t0 = time.time()

    if only is not None:
        steps = [only]
    else:
        steps = [s for s in range(1, 6) if s >= start]

    print("\n" + "#" * 60)
    print("#  PIPELINE DỰ BÁO NGUY CƠ CHÁY TẠI VIỆT NAM")
    print("#  Các bước sẽ chạy:", steps)
    print("#" * 60 + "\n")

    if 1 in steps:
        load_labels()
    if 2 in steps:
        generate_negatives()
    if 3 in steps:
        fetch_features()
        # Bước 3b (tùy chọn): thêm NDVI nếu đã xác thực Google Earth Engine.
        # Chưa xác thực thì tự bỏ qua, pipeline vẫn chạy bình thường.
        try:
            from fetch_ndvi import add_ndvi_to_dataset
            add_ndvi_to_dataset()
        except Exception as e:
            print(f"  (Bỏ qua NDVI: {e})")
    if 4 in steps:
        train_model()
    if 5 in steps:
        make_risk_map()

    dt = time.time() - t0
    print("#" * 60)
    print(f"#  HOÀN TẤT trong {dt:.1f} giây. Xem kết quả trong thư mục outputs/")
    print("#" * 60)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Pipeline dự báo nguy cơ cháy VN")
    ap.add_argument("--step", type=int, default=None,
                    help="Chỉ chạy đúng 1 bước (1..5)")
    ap.add_argument("--from", dest="start", type=int, default=1,
                    help="Chạy từ bước này tới hết (mặc định 1)")
    args = ap.parse_args()
    run_all(start=args.start, only=args.step)
