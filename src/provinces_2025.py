# -*- coding: utf-8 -*-
"""
provinces_2025.py — 34 đơn vị hành chính cấp tỉnh của Việt Nam TỪ 1/7/2025
(6 thành phố trực thuộc TW + 28 tỉnh). Tọa độ đặt tại TRUNG TÂM HÀNH CHÍNH.

Nguồn sáp nhập: Nghị quyết Quốc hội 2025 (63 -> 34: 11 đơn vị giữ nguyên,
52 đơn vị nhập thành 23). Ghi chú () = các tỉnh cũ được nhập vào.
Sinh file web/static/geo/vn_provinces.geojson để hiện nhãn trên bản đồ.
"""
import json
from pathlib import Path

# (tên hiển thị, vĩ độ, kinh độ)
UNITS = [
    # 6 thành phố trực thuộc TW
    ("TP. Hà Nội", 21.03, 105.85),
    ("TP. Hải Phòng", 20.86, 106.68),      # + Hải Dương
    ("TP. Huế", 16.46, 107.59),
    ("TP. Đà Nẵng", 16.05, 108.22),        # + Quảng Nam
    ("TP. Hồ Chí Minh", 10.78, 106.70),    # + Bình Dương + Bà Rịa–Vũng Tàu
    ("TP. Cần Thơ", 10.03, 105.78),        # + Sóc Trăng + Hậu Giang
    # 9 tỉnh giữ nguyên
    ("Cao Bằng", 22.67, 106.26),
    ("Lạng Sơn", 21.85, 106.76),
    ("Lai Châu", 22.39, 103.46),
    ("Điện Biên", 21.39, 103.02),
    ("Sơn La", 21.33, 103.90),
    ("Quảng Ninh", 20.95, 107.08),
    ("Thanh Hóa", 19.81, 105.78),
    ("Nghệ An", 18.79, 105.56),
    ("Hà Tĩnh", 18.34, 105.90),
    # 19 tỉnh mới (sáp nhập)
    ("Tuyên Quang", 22.15, 105.10),        # + Hà Giang
    ("Lào Cai", 21.95, 104.20),            # + Yên Bái
    ("Thái Nguyên", 21.75, 105.90),        # + Bắc Kạn
    ("Phú Thọ", 21.15, 105.20),            # + Vĩnh Phúc + Hòa Bình
    ("Bắc Ninh", 21.25, 106.15),           # + Bắc Giang
    ("Hưng Yên", 20.70, 106.15),           # + Thái Bình
    ("Ninh Bình", 20.25, 105.97),          # + Hà Nam + Nam Định
    ("Quảng Trị", 17.15, 106.60),          # + Quảng Bình
    ("Quảng Ngãi", 15.00, 108.25),         # + Kon Tum
    ("Gia Lai", 13.80, 108.45),            # + Bình Định
    ("Khánh Hòa", 12.20, 109.05),          # + Ninh Thuận
    ("Đắk Lắk", 12.70, 108.30),            # + Phú Yên
    ("Lâm Đồng", 11.75, 108.10),           # + Đắk Nông + Bình Thuận
    ("Tây Ninh", 11.05, 106.15),           # + Long An
    ("Đồng Nai", 11.15, 107.10),           # + Bình Phước
    ("Vĩnh Long", 10.05, 106.15),          # + Bến Tre + Trà Vinh
    ("Đồng Tháp", 10.55, 105.75),          # + Tiền Giang
    ("An Giang", 10.30, 105.10),           # + Kiên Giang
    ("Cà Mau", 9.20, 105.15),              # + Bạc Liêu
]


def build(out_path="web/static/geo/vn_provinces.geojson"):
    fc = {"type": "FeatureCollection", "features": []}
    for name, lat, lon in UNITS:
        city = name.startswith("TP.")
        fc["features"].append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {"name": name, "rank": 0 if city else 1},
        })
    Path(out_path).write_text(json.dumps(fc, ensure_ascii=False))
    print(f"Đã tạo {len(UNITS)} nhãn (34 đơn vị 2025) -> {out_path}")


if __name__ == "__main__":
    build()
