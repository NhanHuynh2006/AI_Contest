# -*- coding: utf-8 -*-
"""
heatmap.py — Dựng LỚP ẢNH NHIỆT MƯỢT (smooth heatmap) cho web, kiểu Windy/Ventusky.

Thay vì vẽ từng ô rời rạc, ta NỘI SUY (interpolate) giá trị nguy cơ của lưới thành
một trường màu liền mạch, chuyển màu êm, rồi phủ lên bản đồ như các web thời tiết
chuyên nghiệp. Ảnh PNG (có độ trong suốt) được đặt đúng theo tọa độ địa lý.
"""

import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors

from config import LAT_MIN, LAT_MAX, LON_MIN, LON_MAX
from predict_service import compute_grid

# Độ phân giải ảnh nhiệt (rộng theo kinh độ x cao theo vĩ độ). Càng lớn càng mịn.
NX, NY = 680, 1180

# Bảng màu tùy biến kiểu "fire risk" chuyên nghiệp: xanh mát (an toàn) chuyển dần
# sang vàng → cam → đỏ → đỏ thẫm (cực nguy hiểm). Rực và tương phản hơn turbo,
# đọc trực giác đúng chuẩn cảnh báo (đỏ = nguy hiểm).
_FIRE_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list("firerisk", [
    (0.00, "#1a3a6b"),   # xanh dương đậm  – rất an toàn
    (0.18, "#1f8fb0"),   # xanh ngọc        – an toàn
    (0.34, "#3fc57a"),   # xanh lá          – thấp
    (0.50, "#c8e05a"),   # vàng chanh       – trung bình
    (0.64, "#ffd23f"),   # vàng             – hơi cao
    (0.78, "#ff8c2b"),   # cam              – cao
    (0.90, "#ff3b2f"),   # đỏ               – rất cao
    (1.00, "#8a0f2e"),   # đỏ thẫm          – cực nguy hiểm
], N=256)


def _interp_field(lon, lat, val, MX, MY):
    """Nội suy 1 trường vô hướng lên lưới ảnh: cubic (mượt) + lấp bằng linear/nearest."""
    Zc = griddata((lon, lat), val, (MX, MY), method="cubic")
    Zl = griddata((lon, lat), val, (MX, MY), method="linear")
    Zn = griddata((lon, lat), val, (MX, MY), method="nearest")
    Z = np.where(~np.isnan(Zc), Zc, np.where(~np.isnan(Zl), Zl, Zn))
    return Z


def render_heatmap(date, step=None, out_path=None, progress_cb=None):
    """
    Tính nguy cơ trên lưới -> nội suy CUBIC mượt, CẮT theo đất liền -> lưu PNG đẹp.
    Trả về dict {bounds, stats}. bounds dạng [[latMin,lonMin],[latMax,lonMax]].
    """
    df = compute_grid(date, step=step, progress_cb=progress_cb)
    pts = df.dropna(subset=["risk"])
    # Không lấy được thời tiết (mạng lỗi / hết lượt gọi dự báo) -> báo rõ ràng
    if len(pts) < 20:
        raise ValueError("Chưa lấy được dữ liệu thời tiết cho ngày này "
                         "(có thể do mạng hoặc tạm hết lượt gọi dự báo). "
                         "Hãy thử lại sau ít phút, hoặc chọn ngày đã có sẵn.")
    lon = pts["longitude"].values
    lat = pts["latitude"].values
    val = pts["risk"].values

    # Lưới ảnh: hàng 0 = phía BẮC (lat lớn nhất) để khớp ảnh trên bản đồ
    gx = np.linspace(LON_MIN, LON_MAX, NX)
    gy = np.linspace(LAT_MAX, LAT_MIN, NY)
    MX, MY = np.meshgrid(gx, gy)

    # --- Trường nguy cơ: nội suy cubic cho gradient hữu cơ, mượt, KHÔNG loang lổ ---
    Z = _interp_field(lon, lat, val, MX, MY)
    Z = gaussian_filter(Z, sigma=1.1)          # làm mượt nhẹ, vẫn giữ chi tiết
    Z = np.clip(Z, 0.0, 1.0)

    # --- MẶT NẠ LÃNH THỔ VIỆT NAM: cắt gọn theo biên giới thật (hình chữ S) ---
    # Lấy từ ranh giới quốc gia GAUL (cache .npy). Nhờ vậy màu KHÔNG loang ra biển
    # và chỉ tô trên đất Việt Nam — vừa đẹp vừa đúng phạm vi mô hình.
    from landmask import get_landmask
    land = get_landmask(shape=(NY, NX))
    if land is None:                                # GEE lỗi -> fallback độ cao
        elev = df["elevation"].fillna(-500).values
        E = _interp_field(df["longitude"].values, df["latitude"].values, elev, MX, MY)
        land = (E >= 0).astype(float)
    land = gaussian_filter(land, sigma=1.4)         # bờ biển mượt, không răng cưa
    # ngưỡng đặt giữa (0.45-0.7) -> mép giữ NGUYÊN vị trí biên giới, không nở ra ngoài
    land = np.clip((land - 0.45) / 0.25, 0.0, 1.0)

    # --- Màu + độ trong suốt kiểu chuyên nghiệp ---
    rgba = _FIRE_CMAP(Z)
    # Vùng an toàn (nguy cơ thấp) TRONG SUỐT sạch sẽ; nguy cơ cao thì RỰC & đậm.
    # Ngưỡng: <10% gần như vô hình, >65% đục hoàn toàn -> mắt bắt ngay điểm nóng.
    a_risk = np.clip((Z - 0.10) / 0.55, 0.0, 1.0) ** 0.85
    rgba[..., 3] = np.clip(a_risk * land, 0.0, 1.0) * 0.94

    if out_path is not None:
        plt.imsave(out_path, rgba)

    # Thống kê từ ô đất liền có dữ liệu
    land = df[(df["elevation"].fillna(-9999) >= 0) & df["risk"].notna()]
    stats = {
        "mean_risk": round(float(land["risk"].mean()), 3) if len(land) else 0.0,
        "max_risk": round(float(land["risk"].max()), 3) if len(land) else 0.0,
        "n_cells": int(len(land)),
        "n_high": int((land["risk"] >= 0.6).sum()),
    }
    # Lưới thô (để web hiện giá trị nguy cơ khi rê chuột — kiểu web thời tiết)
    has_ndvi = "ndvi" in land.columns
    grid_pts = [
        {"lat": round(float(r.latitude), 3), "lon": round(float(r.longitude), 3),
         "risk": round(float(r.risk), 3),
         "t": None if np.isnan(r.temp_max) else round(float(r.temp_max), 1),
         "h": None if np.isnan(r.humidity_mean) else round(float(r.humidity_mean)),
         "w": None if np.isnan(r.wind_max) else round(float(r.wind_max), 1),
         "p": None if np.isnan(r.precip_sum) else round(float(r.precip_sum), 1),
         "n": (None if (not has_ndvi or getattr(r, "ndvi", None) is None
                        or np.isnan(r.ndvi)) else round(float(r.ndvi), 2))}
        for r in land.itertuples(index=False)
    ]
    bounds = [[LAT_MIN, LON_MIN], [LAT_MAX, LON_MAX]]
    return {"bounds": bounds, "stats": stats, "grid": grid_pts}
