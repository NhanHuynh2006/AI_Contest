# -*- coding: utf-8 -*-
"""
plot_3d_risk.py — Biểu đồ CỘT 3D nguy cơ cháy (kiểu surface/bar3d) như bản demo.

Mỗi ô lưới trên lãnh thổ Việt Nam là một cột 3D: CAO theo mức nguy cơ, MÀU theo
mức nguy cơ (xanh dương = an toàn → đỏ = nguy hiểm). Nhìn trực quan "địa hình nguy
cơ" nhô lên ở vùng mùa khô. Xuất PNG để xem/đưa vào báo cáo.

Dùng:
    python src/plot_3d_risk.py 2026-04-15          # 1 ngày
    python src/plot_3d_risk.py 2026-04-15 0.4 35    # ngày, bước lưới, góc xoay
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

from config import LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, OUTPUT_DIR
from predict_service import compute_grid
from landmask import get_landmask
from heatmap import _FIRE_CMAP


def plot_3d(date, step=0.5, azim=-60, elev=38, out_path=None, points=None):
    """
    Vẽ biểu đồ cột 3D cho 1 ngày.
    points: (tuỳ chọn) list dict {lat, lon, risk} đã tính sẵn (vd lấy từ cache
    heatmap) -> KHÔNG cần gọi lại mô hình, rất nhanh. Nếu None thì tự tính.
    """
    # 1) Nguy cơ tại lưới điểm
    if points:
        lat = np.array([p["lat"] for p in points], float)
        lon = np.array([p["lon"] for p in points], float)
        val = np.array([p["risk"] for p in points], float)
    else:
        df = compute_grid(date, step=step)
        pts = df.dropna(subset=["risk"])
        lon, lat, val = pts["longitude"].values, pts["latitude"].values, pts["risk"].values

    # 2) Nội suy lên lưới đều (thô để ra "cột" rõ ràng như ảnh mẫu)
    NX, NY = 46, 92
    gx = np.linspace(LON_MIN, LON_MAX, NX)
    gy = np.linspace(LAT_MIN, LAT_MAX, NY)
    MX, MY = np.meshgrid(gx, gy)
    Z = griddata((lon, lat), val, (MX, MY), method="linear")
    Zn = griddata((lon, lat), val, (MX, MY), method="nearest")
    Z = np.where(np.isnan(Z), Zn, Z)

    # 3) Cắt theo lãnh thổ Việt Nam (mask hàng 0 = BẮC -> lật cho khớp trục lat tăng)
    land = get_landmask(shape=(NY, NX))
    if land is not None:
        land = land[::-1, :]                 # về thứ tự lat TĂNG dần như MY
        Z = np.where(land > 0.5, Z, np.nan)

    # 4) Vẽ cột 3D — khung DỌC (khớp hình chữ S cao của Việt Nam) để đỡ trống 2 bên
    fig = plt.figure(figsize=(7.6, 10.6), facecolor="#0a0f1e")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#0a0f1e")
    # Trục lấp gần hết khung, chừa mép trên cho tiêu đề
    ax.set_position([-0.02, 0.0, 1.04, 0.92])

    dx = (gx[1] - gx[0]) * 0.92
    dy = (gy[1] - gy[0]) * 0.92
    xs, ys, zs, cs = [], [], [], []
    for i in range(NY):
        for j in range(NX):
            z = Z[i, j]
            if np.isnan(z):
                continue
            xs.append(MX[i, j]); ys.append(MY[i, j]); zs.append(max(z, 0.001))
            cs.append(_FIRE_CMAP(float(np.clip(z, 0, 1))))
    xs, ys, zs = np.array(xs), np.array(ys), np.array(zs)
    ax.bar3d(xs, ys, np.zeros_like(zs), dx, dy, zs,
             color=cs, shade=True, edgecolor="none", zsort="max")

    # 5) Thẩm mỹ: nền tối, trục mờ, tiêu đề
    ax.set_title(f"NGUY CƠ CHÁY RỪNG VIỆT NAM · {date}\ncột càng cao & càng đỏ = nguy cơ càng lớn",
                 color="#ffd76b", fontsize=13, pad=6, y=0.99)
    ax.set_zlim(0, 1)
    ax.set_xlabel("Kinh độ", color="#8fa0c2", labelpad=2)
    ax.set_ylabel("Vĩ độ", color="#8fa0c2", labelpad=6)
    ax.set_zlabel("Nguy cơ", color="#8fa0c2", labelpad=2)
    ax.tick_params(colors="#5f7099", labelsize=7, pad=1)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_facecolor((0.04, 0.06, 0.12, 1.0))
        pane.pane.set_edgecolor((0.3, 0.35, 0.5, 0.3))
    ax.view_init(elev=elev, azim=azim)
    # box_aspect: kéo trục VĨ ĐỘ (y) dài ra -> hình dọc, ít khoảng trống 2 bên
    ax.set_box_aspect((NX / NY * 1.25, 2.05, 0.85), zoom=1.28)

    out_path = out_path or str(OUTPUT_DIR / f"risk_3d_{date}.png")
    plt.savefig(out_path, dpi=132, facecolor="#0a0f1e",
                bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Đã lưu: {out_path}  ({len(zs)} cột)")
    return out_path


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else "2026-04-15"
    step = float(sys.argv[2]) if len(sys.argv) > 2 else 0.4
    azim = float(sys.argv[3]) if len(sys.argv) > 3 else -60
    plot_3d(d, step=step, azim=azim)
