# 🔥 Dự báo nguy cơ cháy tại Việt Nam bằng học máy

Dự án cho cuộc thi **AI ứng dụng bảo vệ môi trường**. Bài toán thuộc dạng
**Fire Susceptibility / Risk Prediction** — dự báo *nguy cơ* một nơi có thể cháy
dựa trên điều kiện môi trường (thời tiết, địa hình), **KHÔNG** phải phát hiện
đám cháy đang diễn ra.

> **Ý tưởng cốt lõi:** dữ liệu điểm cháy vệ tinh NASA FIRMS được dùng làm **NHÃN**
> để huấn luyện. Mô hình học quan hệ giữa *điều kiện môi trường* và việc một nơi
> đã từng cháy hay chưa, rồi vẽ **bản đồ nguy cơ**.

---

## 1. Cài đặt

Cần **Python 3.10+**. Cài thư viện:

```bash
pip install -r requirements.txt
```

Dự án này dùng sẵn môi trường ảo tại
`/home/nolan/Documents/Reinforcement_Learning/train`. Nếu dùng nó:

```bash
source /home/nolan/Documents/Reinforcement_Learning/train/bin/activate
```

## 2. Chạy toàn bộ pipeline

```bash
python src/run_pipeline.py
```

Hoặc chạy từng bước:

```bash
python src/run_pipeline.py --step 1     # chỉ bước 1
python src/run_pipeline.py --from 3     # từ bước 3 tới hết
python src/load_labels.py               # cũng có thể chạy trực tiếp từng file
```

## 3. Năm bước của pipeline

| Bước | File | Nhiệm vụ | Kết quả |
|---|---|---|---|
| 1 | `load_labels.py` | Nạp & thống kê **nhãn dương** (119.411 điểm cháy 2020–2026) | in thống kê |
| 2 | `generate_negatives.py` | Sinh **nhãn âm** bằng pseudo-absence sampling vành đai 7–60km + `cKDTree` | `data/negative_points.csv` |
| 3 | `fetch_features.py` | Lấy **đặc trưng môi trường** (thời tiết 14 ngày) + `fetch_ndvi.py` (NDVI từ GEE) | `data/dataset_with_features.csv` |
| 4 | `train_model.py` | Huấn luyện **RF + XGBoost**, kiểm định theo thời gian, SHAP | `outputs/*.png`, `outputs/model.joblib` |
| 5 | `make_risk_map.py` | Dự đoán trên lưới ô, vẽ **bản đồ nguy cơ** | `outputs/risk_map.html` |

## 4. Dữ liệu

- `data/fire_labels_vietnam_2020_2026.csv` — **có sẵn**, **119.411 điểm cháy** thực vật
  đáng tin trong **6.5 năm** (NASA FIRMS MODIS C6.1, đã lọc confidence ≥ 30 và loại
  nguồn nhiệt phi cháy). Tất cả có `label = 1`. Mùa cháy đỉnh **tháng 3–4**.
- `data/negative_points.csv` — **code tự sinh** (bước 2). Điểm nằm trong **vành đai
  7–60 km** quanh điểm cháy (đủ xa để chắc "không cháy", đủ gần để ở LẠI trên đất
  liền — tránh điểm âm rơi xuống biển làm mô hình học mẹo). Gán `label = 0`.
- `data/dataset_with_features.csv` — **code tự tạo** (bước 3). Bảng học máy hoàn chỉnh.

### Đặc trưng dùng để học (10 đặc trưng)
| Đặc trưng | Ý nghĩa | Nguồn |
|---|---|---|
| `temp_max` | Nhiệt độ tối đa NGÀY quan sát (°C) | Open-Meteo |
| `humidity_mean` | Độ ẩm trung bình (%) | Open-Meteo |
| `wind_max` | Tốc độ gió tối đa (km/h) | Open-Meteo |
| `precip_sum` | Lượng mưa ngày quan sát (mm) | Open-Meteo |
| `precip_7d` | **Tổng mưa 7 ngày** — độ ẩm nhiên liệu ngắn hạn | Open-Meteo / ERA5 |
| `precip_14d` | **Tổng mưa 14 ngày** — độ khô tích lũy | Open-Meteo / ERA5 |
| `dry_days_14d` | **Số ngày khô (<1mm)/14 ngày** — chuỗi ngày hạn | Open-Meteo / ERA5 |
| `elevation` | Độ cao địa hình (m) | Open-Meteo |
| `month` | Tháng (nắm bắt tính mùa vụ) | từ ngày quan sát |
| `ndvi` | **Thảm thực vật NDVI** — biến dự báo mạnh nhất theo nghiên cứu VN | MODIS/Google Earth Engine |

> **Vì sao mạnh hơn bản đầu:** thêm đặc trưng "độ khô tích lũy" (mưa 7/14 ngày,
> chuỗi ngày hạn) và NDVI — đúng theo các bài báo cháy rừng 2024–2025 (chỉ số ẩm
> nhiên liệu FFMC/DC & NDVI là yếu tố dự báo hàng đầu). Mô hình được **kiểm định
> theo thời gian** (train 2020–2024 → test 2025–2026) và có biểu đồ **SHAP** giải
> thích chiều tác động từng đặc trưng.

## 5. Cấu hình nhanh — `src/config.py`

- `SAMPLE_PER_CLASS = 3000` — số điểm mỗi lớp lấy để **demo cho nhanh** (do API
  thời tiết miễn phí giới hạn ~10k điểm/ngày). Đặt `= None` để chạy **toàn bộ**.
- `MIN_DISTANCE_KM = 7.0`, `MAX_DISTANCE_KM = 60.0` — vành đai điểm âm quanh điểm cháy.
- `GRID_STEP_DEG = 0.25` — độ mịn của lưới bản đồ (nhỏ hơn = mịn hơn, chậm hơn).
- `RISK_MAP_DATE = "2026-04-15"` — ngày đại diện (mùa khô) để vẽ bản đồ.

## 6. Vì sao gọi API không bị chậm/lỗi?

- **Gom theo ngày + gửi nhiều tọa độ/ 1 request** → ~4000 điểm chỉ còn ~200 request.
- **Cache** ra `data/cache/weather_cache.csv` → chạy lại gần như tức thì.
- **Retry + delay** → 1 vài điểm lỗi API không làm hỏng cả pipeline.

## 7. Kết quả mong đợi

- `outputs/roc_curve.png` — đường cong ROC. **AUC ≈ 0.80** (chia ngẫu nhiên),
  **AUC ≈ 0.76** khi kiểm định theo thời gian (train 2020–2024 → test 2025–2026).
  > Thấp hơn con số 0.9x của bản đầu vì điểm âm giờ **khó hơn** (vành đai 7–60km,
  > trùng phân phối ngày) — mô hình không còn "ăn gian" bằng mẹo biển/mùa, nên con
  > số này **trung thực và đáng tin hơn**.
- `outputs/confusion_matrix.png` — ma trận nhầm lẫn.
- `outputs/feature_importance.png` — đặc trưng nào quyết định nguy cơ.
- `outputs/shap_summary.png` — **SHAP**: mỗi đặc trưng đẩy nguy cơ LÊN/XUỐNG thế nào.
- `outputs/risk_map.html` — bản đồ nguy cơ phủ màu, mở bằng trình duyệt.

## 8. 🌐 Web demo — "VN FireWatch"

Web demo dựng bằng **MapLibre GL (WebGL)**. Mặc định **bản đồ nhiệt 2D** cắt đúng
**ranh giới Việt Nam** (geoBoundaries, ĐH William & Mary), có tên **34 tỉnh/thành
(2025)**, lưới kinh/vĩ độ, và **Hoàng Sa & Trường Sa** 🇻🇳. Bấm nút **3D** để xem
**biểu đồ cột 3D** nguy cơ (mỗi ô 1 cột, cao & đỏ = nguy cơ cao) theo đúng ngày đang xem.

```bash
python web/app.py
# rồi mở trình duyệt: http://127.0.0.1:5000
```

**Tính năng:**
- 🗺️ **Bản đồ nhiệt 2D** nội suy mượt (cubic), cắt gọn theo biên giới thật.
- 🧊 **Biểu đồ cột 3D** (matplotlib) theo ngày — nút **3D**, có **zoom + kéo**.
- 🏷️ **34 tỉnh/thành 2025** (chấm + tên), 📐 lưới kinh/vĩ độ (bật/tắt được).
- 🖱️ **Rê chuột đọc số liệu**: nguy cơ % + nhiệt độ/độ ẩm/gió/mưa + NDVI tại điểm.
- 🇻🇳 **Quần đảo Hoàng Sa & Trường Sa** của Việt Nam.
- 📅 **Chọn ngày bất kỳ** (2000→nay): lần đầu ~10–15s, sau đó tức thì nhờ cache.

**KHÔNG PHỤ THUỘC QUOTA (quan trọng cho ngày thi):**
- Thời tiết lấy thẳng từ **ERA5-Land qua Google Earth Engine** (`src/fetch_weather_gee.py`)
  — MIỄN PHÍ, không giới hạn ngặt như Open-Meteo. Bật/tắt bằng `USE_GEE_WEATHER`
  trong `src/config.py` (mặc định `True`). Open-Meteo archive vốn là ERA5 nên giá trị
  gần như y hệt.
- Đã **pre-cache** sẵn nhiều ngày mùa khô 2023–2026 (heatmap + ảnh 3D) trong
  `web/static/heatmaps/` và `web/static/plots3d/` → demo bấm là hiện ngay.

**Bên trong (điểm kỹ thuật để trình bày):**
- `src/heatmap.py`: nội suy `scipy.griddata` (cubic) → bảng màu "firerisk" →
  **cắt theo mặt nạ lãnh thổ Việt Nam** (`src/landmask.py`) → PNG trong suốt.
- `src/plot_3d_risk.py`: biểu đồ `bar3d` (matplotlib), tái dùng lưới đã cache của
  heatmap nên dựng <1s.
- Nguồn dữ liệu (đều chuẩn quốc tế): NASA FIRMS (cháy), ERA5-Land/Copernicus
  (thời tiết), MODIS (NDVI), FAO/geoBoundaries (ranh giới) — qua Google Earth Engine.

**So sánh mùa để thuyết trình:** mở 15/04 (đỏ rực – mùa khô) rồi 15/06 (xanh nhiều
hơn – mùa mưa) để thấy mô hình nắm được tính mùa vụ.

## 9. Cấu trúc thư mục

```
AI_contest/
├── data/
│   ├── fire_labels_vietnam.csv       # nhãn dương (có sẵn)
│   ├── fire_data_raw_vietnam.csv     # dữ liệu gốc (tham khảo)
│   ├── negative_points.csv           # sinh ở bước 2
│   ├── dataset_with_features.csv     # sinh ở bước 3
│   └── cache/weather_cache.csv       # cache thời tiết
├── src/
│   ├── config.py            # cấu hình chung
│   ├── weather_api.py       # gọi Open-Meteo (batch + cache + retry)
│   ├── load_labels.py       # bước 1
│   ├── generate_negatives.py# bước 2
│   ├── fetch_features.py    # bước 3
│   ├── train_model.py       # bước 4
│   ├── make_risk_map.py     # bước 5
│   ├── predict_service.py   # dịch vụ dự đoán lưới cho web
│   ├── heatmap.py           # nội suy -> ảnh nhiệt mượt (smooth heatmap)
│   └── run_pipeline.py      # chạy tất cả
├── web/
│   ├── app.py               # server Flask (/, /api/heatmap, /api/timelapse, /api/info)
│   ├── templates/index.html # giao diện bản đồ nhiệt + hoạt ảnh
│   └── static/heatmaps/     # ảnh nhiệt đã dựng (cache theo ngày)
├── outputs/                 # biểu đồ + mô hình + bản đồ
├── requirements.txt
└── README.md
```

cd /home/nolan/Documents/AI_contest && /home/nolan/Documents/Reinforcement_Learning/train/bin/python web/app.py

cd /home/nolan/Documents/AI_contest && nohup /home/nolan/Documents/Reinforcement_Learning/train/bin/python web/app.py > .logs/web.log 2>&1 &
