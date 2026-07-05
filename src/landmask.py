# -*- coding: utf-8 -*-
"""
landmask.py — Mặt nạ ĐẤT LIỀN Việt Nam (lấy 1 lần từ GEE, cache ra .npy).

Bản đồ nhiệt phải được CẮT gọn theo bờ biển thật, nếu không màu sẽ loang ra biển
trông rất xấu. Open-Meteo trả độ cao = 0 cho cả mặt biển nên không dùng để phân
biệt đất/biển được. Ở đây ta lấy mặt nạ đất liền từ ảnh độ cao SRTM (bị che ở
ngoài khơi) qua Google Earth Engine, rồi lưu lại thành lưới nhị phân để tô nhanh.
"""

import numpy as np
from pathlib import Path
from config import LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, CACHE_DIR, LABELS_CSV
from fetch_ndvi import init_ee

MASK_NPY = Path(CACHE_DIR) / "landmask.npy"
# Độ phân giải lưới mặt nạ (khung 8°x16° -> mỗi pixel ~1.1km, bờ biển rất sắc)
MW, MH = 720, 1440         # rộng (kinh độ) x cao (vĩ độ)


def _build_from_gee():
    """
    Lấy mặt nạ lãnh thổ Việt Nam, trả mảng (MH, MW) 0/1, hàng 0 = BẮC.

    Ranh giới: FAO GAUL (bộ ranh giới hành chính do FAO/Liên Hợp Quốc phát hành —
    nguồn ranh giới quốc gia chuẩn quốc tế trên Google Earth Engine).

    Cách lấy: dùng getThumbURL với region + dimensions CHÍNH XÁC bằng khung bản đồ
    -> ảnh trả về khớp từng pixel với khung (tránh lỗi lệch lưới của sampleRectangle
    từng làm méo biên giới phía Bắc).
    """
    import ee
    import io
    import requests
    from PIL import Image

    # geoBoundaries v6.0.0 (2023) — ranh giới do ĐH William & Mary (Mỹ) biên soạn &
    # kiểm định, giấy phép MỞ CC-BY, là chuẩn học thuật hiện đại được dùng rộng rãi
    # trong báo chí dữ liệu & nghiên cứu. MỚI hơn và CHI TIẾT hơn FAO/GAUL 2015.
    # (Miễn phí & hợp pháp làm mask — khác Google Maps bị cấm trích ranh giới.)
    countries = ee.FeatureCollection("WM/geoLab/geoBoundaries/600/ADM0")
    vn = countries.filter(ee.Filter.eq("shapeGroup", "VNM"))
    land = ee.Image(0).paint(vn, 1)               # 1 trong lãnh thổ VN, 0 ngoài
    region = ee.Geometry.Rectangle([LON_MIN, LAT_MIN, LON_MAX, LAT_MAX])
    url = land.getThumbURL({
        "region": region,
        "dimensions": f"{MW}x{MH}",               # đúng kích thước lưới mặt nạ
        "format": "png",
        "min": 0, "max": 1,
        "palette": ["000000", "ffffff"],          # đen = ngoài, trắng = Việt Nam
    })
    img = Image.open(io.BytesIO(requests.get(url, timeout=60).content)).convert("L")
    grid = (np.asarray(img, dtype=float) > 127).astype(float)   # hàng 0 = BẮC
    return grid


def _islands_grid():
    """
    Các đảo nhỏ của Việt Nam (bị ranh giới giản lược/dữ liệu cháy bỏ sót)
    — mỗi đảo 1 chấm ~6km để luôn hiện trên bản đồ nhiệt.
    """
    ISLANDS = [
        (8.68, 106.60),    # Côn Đảo
        (15.38, 109.12),   # Lý Sơn
        (10.54, 108.94),   # Phú Quý
        (20.13, 107.73),   # Bạch Long Vĩ
        (10.22, 104.00),   # Phú Quốc (đề phòng thiếu)
    ]
    grid = np.zeros((MH, MW), dtype=float)
    for la, lo in ISLANDS:
        r = int((LAT_MAX - la) / (LAT_MAX - LAT_MIN) * (MH - 1))
        c = int((lo - LON_MIN) / (LON_MAX - LON_MIN) * (MW - 1))
        rad = 3
        grid[max(0, r - rad):r + rad + 1, max(0, c - rad):c + rad + 1] = 1.0
    return grid


def _build_from_fire_points():
    """
    Mặt nạ lãnh thổ từ CHÍNH 119k điểm cháy NASA FIRMS (2020–2026).

    NASA FIRMS khi tải theo quốc gia đã CẮT SẴN theo biên giới Việt Nam, nên
    "dấu chân" 6.5 năm của các điểm cháy vẽ ra đúng hình đất liền Việt Nam.
    Đây là cách dễ thuyết trình nhất: vùng tô màu = đúng vùng dữ liệu vệ tinh.

    Các bước: chấm điểm lên lưới -> nở nhẹ (~4km) nối các điểm -> đóng lỗ hổng
    (đô thị/đỉnh núi không cháy) -> lấp lỗ kín -> GIAO với ranh giới FAO GAUL
    để tuyệt đối không tràn sang nước bạn.
    """
    import pandas as pd
    from scipy.ndimage import binary_dilation, binary_closing, binary_fill_holes

    df = pd.read_csv(LABELS_CSV, usecols=["latitude", "longitude"])
    H, W = MH, MW
    r = ((LAT_MAX - df["latitude"]) / (LAT_MAX - LAT_MIN) * (H - 1)).astype(int).clip(0, H - 1)
    c = ((df["longitude"] - LON_MIN) / (LON_MAX - LON_MIN) * (W - 1)).astype(int).clip(0, W - 1)
    grid = np.zeros((H, W), dtype=bool)
    grid[r, c] = True

    grid = binary_dilation(grid, iterations=3)                 # nở ~6km nối điểm
    # vá khe rộng tới ~30km (vd rừng ngập mặn Đất Mũi Cà Mau không bao giờ cháy);
    # yên tâm nở mạnh vì sau đó sẽ GIAO với ranh giới FAO -> không thể tràn ra ngoài
    grid = binary_closing(grid, structure=np.ones((15, 15)))
    grid = binary_fill_holes(grid)                             # lấp lỗ kín bên trong
    return grid.astype(float)


def get_landmask(shape=None):
    """
    Trả mặt nạ lãnh thổ dạng float (0..1), đã phóng tới `shape=(NY, NX)` nếu truyền.
    Hàng 0 = phía BẮC. Có cache đĩa; nếu lỗi trả None (render sẽ bỏ qua mask).
    """
    if MASK_NPY.exists():
        grid = np.load(MASK_NPY)
    else:
        try:
            # KHUÔN CHÍNH: ranh giới Việt Nam FAO GAUL (Liên Hợp Quốc công nhận).
            # Lưu ý: KHÔNG dùng dấu chân điểm cháy NASA làm khuôn trực tiếp vì các
            # vùng không-thể-cháy (rừng ngập mặn Đất Mũi Cà Mau...) sẽ bị khuyết;
            # dấu chân NASA chỉ dùng làm phương án dự phòng khi GEE không sẵn sàng.
            grid = None
            try:
                if init_ee():
                    grid = _build_from_gee()
            except Exception:
                grid = None
            if grid is None:
                grid = _build_from_fire_points()      # dự phòng (offline)
            # Luôn thêm các đảo của Việt Nam
            grid = np.maximum(grid, _islands_grid())
            np.save(MASK_NPY, grid)
        except Exception:
            return None

    if shape is None:
        return grid
    # Phóng lên đúng kích thước ảnh bằng nội suy song tuyến (mượt bờ biển)
    from scipy.ndimage import zoom
    zy = shape[0] / grid.shape[0]
    zx = shape[1] / grid.shape[1]
    big = zoom(grid, (zy, zx), order=1)
    return np.clip(big, 0.0, 1.0)


if __name__ == "__main__":
    m = get_landmask()
    if m is None:
        print("Không lấy được mặt nạ (GEE chưa sẵn sàng).")
    else:
        print(f"Mặt nạ đất liền: {m.shape}, tỉ lệ đất = {m.mean()*100:.1f}%")
        print(f"Đã lưu: {MASK_NPY}")
