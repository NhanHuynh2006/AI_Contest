# -*- coding: utf-8 -*-
"""
Bước 1 — load_labels.py: Nạp NHÃN DƯƠNG (các điểm đã từng cháy).

Đọc fire_labels_vietnam.csv (NASA FIRMS MODIS, đã làm sạch) và in ra thống kê
để hiểu dữ liệu: số điểm, khoảng thời gian, phân bố theo tháng.
Các điểm này đều có label = 1 (đã từng cháy).
"""

import pandas as pd

from config import LABELS_CSV


def load_labels(verbose=True):
    """Đọc file nhãn dương, trả về DataFrame. In thống kê nếu verbose=True."""
    df = pd.read_csv(LABELS_CSV)

    if verbose:
        print("=" * 60)
        print("BƯỚC 1: NẠP NHÃN DƯƠNG (điểm đã từng cháy)")
        print("=" * 60)
        print(f"  File: {LABELS_CSV.name}")
        print(f"  Tổng số điểm cháy: {len(df):,}")
        print(f"  Khoảng thời gian : {df['acq_date'].min()} → {df['acq_date'].max()}")
        print(f"  Số cột đặc trưng : {df.shape[1]}")
        print()
        print("  Phân bố theo tháng:")
        monthly = df["month"].value_counts().sort_index()
        tong = monthly.sum()
        for thang, so_diem in monthly.items():
            ty_le = so_diem / tong * 100
            thanh = "#" * int(ty_le / 2)      # thanh bar đơn giản bằng ký tự
            print(f"    Tháng {thang:>2}: {so_diem:>6,}  ({ty_le:4.1f}%)  {thanh}")
        print()
        print(f"  Vùng bao (bounding box):")
        print(f"    Vĩ độ  (lat): {df['latitude'].min():.2f} → {df['latitude'].max():.2f}")
        print(f"    Kinh độ(lon): {df['longitude'].min():.2f} → {df['longitude'].max():.2f}")
        print()

    return df


if __name__ == "__main__":
    load_labels()
