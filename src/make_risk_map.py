# -*- coding: utf-8 -*-
"""
Bước 5 — make_risk_map.py: Tạo BẢN ĐỒ NGUY CƠ CHÁY (HTML tương tác).

- Phủ một lưới ô đều lên lãnh thổ Việt Nam (mỗi ô GRID_STEP_DEG độ)
- Lấy đặc trưng môi trường cho TÂM mỗi ô vào một ngày đại diện (khô nóng mùa khô)
- Dùng mô hình đã train để dự đoán XÁC SUẤT nguy cơ cháy cho từng ô
- Vẽ bản đồ folium phủ màu: xanh (an toàn) → vàng → đỏ (nguy cơ cao)
- Xuất risk_map.html mở được trên trình duyệt
"""

import numpy as np
import pandas as pd
import joblib
import folium
from tqdm import tqdm

from config import (
    LAT_MIN, LAT_MAX, LON_MIN, LON_MAX,
    GRID_STEP_DEG, RISK_MAP_DATE, FEATURE_COLUMNS,
    MODEL_FILE, RISK_MAP_HTML,
)
from weather_api import load_weather_cache, fetch_weather_batched


def _build_grid():
    """Sinh lưới tâm ô đều phủ khung Việt Nam. Trả về DataFrame lat/lon/date/month."""
    lats = np.arange(LAT_MIN, LAT_MAX + 1e-9, GRID_STEP_DEG)
    lons = np.arange(LON_MIN, LON_MAX + 1e-9, GRID_STEP_DEG)
    grid_lat, grid_lon = np.meshgrid(lats, lons)
    month = int(RISK_MAP_DATE.split("-")[1])
    return pd.DataFrame({
        "latitude": np.round(grid_lat.ravel(), 4),
        "longitude": np.round(grid_lon.ravel(), 4),
        "acq_date": RISK_MAP_DATE,
        "month": month,
    })


def _risk_color(p):
    """Chuyển xác suất (0..1) thành màu xanh→vàng→đỏ."""
    if p < 0.2:
        return "#2c7bb6"      # xanh dương - rất an toàn
    if p < 0.4:
        return "#abd9e9"      # xanh nhạt
    if p < 0.6:
        return "#ffffbf"      # vàng - trung bình
    if p < 0.8:
        return "#fdae61"      # cam - cao
    return "#d7191c"          # đỏ - rất cao


def make_risk_map(verbose=True):
    """Tạo bản đồ nguy cơ và lưu ra RISK_MAP_HTML."""
    if verbose:
        print("=" * 60)
        print("BƯỚC 5: TẠO BẢN ĐỒ NGUY CƠ CHÁY")
        print("=" * 60)

    # Nạp mô hình đã train (kèm danh sách cột & median để điền giá trị thiếu)
    bundle = joblib.load(MODEL_FILE)
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]
    medians = bundle["medians"]

    grid = _build_grid()
    if verbose:
        print(f"  Mô hình dùng: {bundle.get('model_name', '?')}")
        print(f"  Lưới: {len(grid):,} ô (bước {GRID_STEP_DEG}°)")
        print(f"  Ngày đại diện: {RISK_MAP_DATE}")
        print(f"  Lấy thời tiết cho tâm mỗi ô (có cache)...")

    # Lấy đặc trưng cho tâm các ô (cùng 1 ngày -> gom lô rất nhanh)
    cache = load_weather_cache()
    if verbose:
        bar = tqdm(total=len(grid), desc="  Lấy thời tiết", ncols=80)
        cb = bar.update
    else:
        cb = None
    feats = fetch_weather_batched(grid[["latitude", "longitude", "acq_date"]],
                                  cache, progress_cb=cb)
    if verbose:
        bar.close()

    for col in feats.columns:
        grid[col] = feats[col].values

    # Chuẩn bị ma trận đặc trưng, điền giá trị thiếu bằng median lúc train
    X = grid[feature_columns].copy()
    X = X.fillna(medians)

    # Dự đoán xác suất nguy cơ cháy
    grid["risk"] = model.predict_proba(X)[:, 1]

    # Bỏ ô trên biển/không có dữ liệu độ cao rõ ràng để bản đồ sạch hơn:
    # giữ ô có độ cao >= 0 (đất liền). Điểm biển thường elevation <= 0.
    on_land = grid["elevation"].fillna(-9999) >= 0
    grid_land = grid[on_land]

    if verbose:
        print(f"\n  Số ô trên đất liền vẽ lên bản đồ: {len(grid_land):,}")
        print(f"  Nguy cơ trung bình: {grid_land['risk'].mean():.3f} | "
              f"cao nhất: {grid_land['risk'].max():.3f}")

    # --- Vẽ bản đồ folium ---
    center = [(LAT_MIN + LAT_MAX) / 2, (LON_MIN + LON_MAX) / 2]
    fmap = folium.Map(location=center, zoom_start=6, tiles="CartoDB positron")

    half = GRID_STEP_DEG / 2.0
    for row in grid_land.itertuples(index=False):
        color = _risk_color(row.risk)
        bounds = [[row.latitude - half, row.longitude - half],
                  [row.latitude + half, row.longitude + half]]
        folium.Rectangle(
            bounds=bounds,
            color=None, weight=0,
            fill=True, fill_color=color, fill_opacity=0.55,
            popup=(f"Nguy cơ: {row.risk:.0%}<br>"
                   f"Nhiệt độ: {row.temp_max}°C<br>"
                   f"Độ ẩm: {row.humidity_mean}%<br>"
                   f"Gió: {row.wind_max} km/h<br>"
                   f"Mưa: {row.precip_sum} mm"),
        ).add_to(fmap)

    # Chú giải màu (legend) đơn giản
    legend_html = """
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999;
                background: white; padding: 10px 14px; border: 1px solid #999;
                border-radius: 6px; font-size: 13px; font-family: sans-serif;">
      <b>Nguy cơ cháy</b><br>
      <span style="color:#d7191c;">&#9632;</span> Rất cao (80-100%)<br>
      <span style="color:#fdae61;">&#9632;</span> Cao (60-80%)<br>
      <span style="color:#ffffbf;">&#9632;</span> Trung bình (40-60%)<br>
      <span style="color:#abd9e9;">&#9632;</span> Thấp (20-40%)<br>
      <span style="color:#2c7bb6;">&#9632;</span> Rất thấp (0-20%)<br>
      <i>Ngày: __DATE__</i>
    </div>
    """.replace("__DATE__", RISK_MAP_DATE)
    fmap.get_root().html.add_child(folium.Element(legend_html))

    fmap.save(str(RISK_MAP_HTML))
    if verbose:
        print(f"  Đã lưu bản đồ: {RISK_MAP_HTML.name}")
        print(f"  Mở bằng trình duyệt để xem bản đồ tương tác.")
        print()


if __name__ == "__main__":
    make_risk_map()
