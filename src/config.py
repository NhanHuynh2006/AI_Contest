# -*- coding: utf-8 -*-
"""
config.py — Cấu hình dùng chung cho toàn bộ pipeline dự báo nguy cơ cháy.

Mọi đường dẫn và tham số quan trọng gom về đây để dễ chỉnh, KHÔNG hard-code
đường dẫn tuyệt đối ở các file khác. Tất cả đường dẫn tính tương đối theo vị
trí thư mục gốc của project (thư mục chứa data/, src/, outputs/).
"""

from pathlib import Path

# ---- Nguồn dữ liệu thời tiết cho WEB ----
# True  = lấy thẳng từ ERA5-Land qua Google Earth Engine (MIỄN PHÍ, không quota).
# False = dùng Open-Meteo (nhanh hơn khi còn lượt, nhưng giới hạn quota/ngày).
# Để True cho an toàn ngày thi — không lo hết lượt gọi API.
USE_GEE_WEATHER = True

# ---- Đường dẫn thư mục (tự suy ra, không hard-code tuyệt đối) ----
# File này nằm trong src/, nên thư mục gốc project là cha của src/.
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"
CACHE_DIR = DATA_DIR / "cache"          # nơi lưu cache thời tiết

# Tạo sẵn các thư mục nếu chưa có (an toàn khi chạy lại nhiều lần)
for _d in (DATA_DIR, OUTPUT_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---- Các file dữ liệu ----
# Bộ nhãn 6.5 năm (2020–2026): 119.411 điểm cháy sạch — bộ chính để train
LABELS_CSV = DATA_DIR / "fire_labels_vietnam_2020_2026.csv"
NEGATIVES_CSV = DATA_DIR / "negative_points.csv"          # sinh ở bước 2
DATASET_CSV = DATA_DIR / "dataset_with_features.csv"      # sinh ở bước 3
# Cache v2: mỗi điểm lưu cả CHUỖI 14 NGÀY thời tiết trước đó (khác v1 chỉ 1 ngày)
WEATHER_CACHE_CSV = CACHE_DIR / "weather_cache_v2.csv"

# ---- Các file kết quả ----
MODEL_FILE = OUTPUT_DIR / "model.joblib"
ROC_PNG = OUTPUT_DIR / "roc_curve.png"
CM_PNG = OUTPUT_DIR / "confusion_matrix.png"
FI_PNG = OUTPUT_DIR / "feature_importance.png"
SHAP_PNG = OUTPUT_DIR / "shap_summary.png"
RISK_MAP_HTML = OUTPUT_DIR / "risk_map.html"

# ---- Khung địa lý Việt Nam ----
LAT_MIN, LAT_MAX = 8.0, 24.0
LON_MIN, LON_MAX = 102.0, 110.0

# ---- Khoảng thời gian dữ liệu (khớp với fire_labels_vietnam_2020_2026.csv) ----
DATE_START = "2020-01-01"
DATE_END = "2026-07-01"

# ---- Tham số sinh điểm âm (pseudo-absence) ----
# Mỗi điểm âm phải cách MỌI điểm cháy ít nhất MIN_DISTANCE_KM (đảm bảo "không cháy")
# NHƯNG cũng phải cách một điểm cháy nào đó KHÔNG QUÁ MAX_DISTANCE_KM.
# Vành đai 7–60km này giữ điểm âm ở LẠI TRÊN ĐẤT LIỀN (cháy chỉ xảy ra trên đất)
# và trong môi trường tương đồng — tránh lỗi điểm âm rơi xuống biển làm mô hình
# học mẹo "biển thì không cháy" (đã phát hiện ở phiên bản đầu).
MIN_DISTANCE_KM = 7.0
MAX_DISTANCE_KM = 60.0
NEG_POS_RATIO = 1.0            # tỉ lệ điểm âm : điểm dương (1.0 = 1:1)

# ---- Tham số lấy đặc trưng (Bước 3) ----
# Gọi API cho ~240.000 điểm là bất khả thi (giới hạn ~10k điểm/ngày của Open-Meteo).
# Lấy mẫu phân tầng theo thời gian. Đặt None để chạy toàn bộ (nhiều ngày!).
SAMPLE_PER_CLASS = 3000        # số điểm mỗi lớp (dương/âm) đưa vào huấn luyện
API_DELAY_SEC = 0.15           # nghỉ giữa các request để tránh bị chặn (rate-limit)
API_MAX_RETRY = 5              # số lần thử lại khi request lỗi (kể cả khi bị rate-limit)
API_TIMEOUT_SEC = 40

# Số ngày thời tiết LỊCH SỬ lấy kèm trước ngày quan sát (tính "độ khô tích lũy").
# 13 ngày trước + ngày quan sát = cửa sổ 14 ngày.
PAST_DAYS = 13

# Các cột đặc trưng thời tiết (đúng thứ tự dùng để train mô hình)
WEATHER_FEATURES = [
    "temp_max",       # nhiệt độ tối đa NGÀY quan sát (°C)
    "humidity_mean",  # độ ẩm trung bình ngày quan sát (%)
    "wind_max",       # tốc độ gió tối đa ngày quan sát (km/h)
    "precip_sum",     # lượng mưa ngày quan sát (mm)
    "precip_7d",      # tổng mưa 7 ngày gần nhất (mm) — độ ẩm nhiên liệu ngắn hạn
    "precip_14d",     # tổng mưa 14 ngày gần nhất (mm) — độ khô tích lũy
    "dry_days_14d",   # số ngày khô (<1mm mưa) trong 14 ngày — chuỗi ngày hạn
]
# Đặc trưng bổ sung (không cần API thời tiết): độ cao & tháng
EXTRA_FEATURES = ["elevation", "month"]
FEATURE_COLUMNS = WEATHER_FEATURES + EXTRA_FEATURES

# ---- Tham số bản đồ nguy cơ (Bước 5) ----
GRID_STEP_DEG = 0.25          # kích thước ô lưới (độ). ~0.1 mịn hơn nhưng nhiều ô hơn.
# Ngày đại diện để lấy thời tiết cho bản đồ (một ngày khô nóng điển hình mùa khô).
RISK_MAP_DATE = "2026-04-15"

# Hạt giống ngẫu nhiên để kết quả lặp lại được
RANDOM_SEED = 42
