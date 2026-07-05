# PROJECT: Dự báo nguy cơ cháy tại Việt Nam bằng học máy

## Bối cảnh
Đây là project cho cuộc thi AI ứng dụng bảo vệ môi trường (học sinh THPT). Bài toán thuộc dạng **Fire Susceptibility / Risk Prediction** (dự báo *nguy cơ* cháy theo điều kiện môi trường), KHÔNG phải phát hiện đám cháy đang xảy ra.

Nguyên tắc cốt lõi: dữ liệu điểm cháy vệ tinh (NASA FIRMS) được dùng làm **nhãn (label)** để huấn luyện, không phải làm sản phẩm cuối. Mô hình học quan hệ giữa điều kiện môi trường (thời tiết, thảm thực vật, địa hình) và việc một nơi có từng cháy hay không, rồi tạo ra bản đồ nguy cơ.

## Mục tiêu sản phẩm
1. Một pipeline Python hoàn chỉnh, chạy được từ đầu đến cuối
2. Mô hình phân loại (Random Forest, có so sánh thêm XGBoost) dự đoán xác suất nguy cơ cháy
3. Bản đồ nguy cơ cháy phủ màu (HTML tương tác bằng folium)
4. Các biểu đồ đánh giá: ROC curve, confusion matrix, feature importance

## Yêu cầu kỹ thuật chung
- Ngôn ngữ: Python 3
- Thư viện: pandas, numpy, scikit-learn, xgboost, matplotlib, folium, requests
- Code chia thành các module/hàm rõ ràng, có comment tiếng Việt
- Mỗi bước in ra thông tin tiến độ (số dòng, kết quả trung gian) để học sinh dễ theo dõi
- Có file `requirements.txt`
- Có `README.md` hướng dẫn chạy

---

## DỮ LIỆU ĐẦU VÀO

### 1. Nhãn dương — điểm đã cháy (ĐÃ CÓ SẴN)
File: `fire_labels_vietnam.csv` (đã được làm sạch sẵn, đính kèm project)
- 12.644 điểm cháy đáng tin trên toàn Việt Nam, từ 01/01/2026 đến 01/07/2026
- Nguồn: NASA FIRMS MODIS C6.1 (đã lọc confidence >= 30, đã loại nguồn nhiệt phi cháy-thực-vật)
- Các cột: `latitude, longitude, acq_date, month, acq_time, daynight, confidence, brightness, bright_t31, frp, satellite, instrument, type, source_file, label`
- Cột `label` = 1 (tất cả đều là điểm dương)

### 2. Nhãn âm — điểm không cháy (CODE CẦN TỰ SINH)
Không có sẵn. Code phải sinh ngẫu nhiên (pseudo-absence sampling):
- Sinh các điểm ngẫu nhiên trong khung Việt Nam (lat 8–24, lon 102–110)
- ĐIỀU KIỆN: mỗi điểm âm phải cách MỌI điểm cháy ít nhất một khoảng tối thiểu (ví dụ 5–10km) để đảm bảo thực sự "không cháy"
- Gán `label` = 0
- Số điểm âm xấp xỉ số điểm dương (tỉ lệ 1:1)
- Gán cho mỗi điểm âm một ngày ngẫu nhiên trong cùng khoảng thời gian dữ liệu (để lấy thời tiết tương ứng)

### 3. Đặc trưng môi trường (CODE CẦN LẤY QUA API)
Với CẢ điểm dương lẫn điểm âm, lấy giá trị tại đúng (tọa độ, ngày):

**a) Thời tiết — API Open-Meteo (ưu tiên, KHÔNG cần API key)**
- Endpoint historical: `https://archive-api.open-meteo.com/v1/archive`
- Tham số: `latitude, longitude, start_date, end_date, daily=temperature_2m_max,relative_humidity_2m_mean,wind_speed_10m_max,precipitation_sum`
- Lấy: nhiệt độ tối đa, độ ẩm trung bình, tốc độ gió tối đa, tổng lượng mưa trong ngày đó
- LƯU Ý: gọi API cho hàng nghìn điểm sẽ chậm và có thể bị rate-limit. Cần:
  - Thêm delay giữa các request
  - Cache kết quả (lưu ra file trung gian để không gọi lại)
  - Xử lý lỗi/timeout gracefully (retry hoặc bỏ qua điểm lỗi)

**b) NDVI (thảm thực vật) — TÙY CHỌN, phần nâng cao**
- Nguồn: Google Earth Engine (cần tài khoản). Nếu triển khai, dùng thư viện `earthengine-api`
- Nếu chưa có GEE, tạo phần này thành module riêng có thể bật/tắt, để mô hình vẫn chạy được chỉ với thời tiết

**c) Địa hình (độ cao, độ dốc) — TÙY CHỌN, phần nâng cao**
- Nguồn: Open-Meteo Elevation API hoặc SRTM qua GEE
- Cũng để dạng module bật/tắt

---

## CÁC BƯỚC PIPELINE (viết thành các hàm/script riêng)

### Bước 1: `load_labels.py` — Nạp nhãn dương
- Đọc `fire_labels_vietnam.csv`
- In thống kê: số điểm, khoảng thời gian, phân bố theo tháng

### Bước 2: `generate_negatives.py` — Sinh nhãn âm
- Sinh điểm ngẫu nhiên thỏa điều kiện khoảng cách tối thiểu tới điểm cháy
- Dùng cấu trúc dữ liệu không gian (ví dụ scipy.spatial.cKDTree) để kiểm tra khoảng cách nhanh, tránh vòng lặp O(n²)
- Gán ngày ngẫu nhiên + label=0
- Xuất `negative_points.csv`

### Bước 3: `fetch_features.py` — Lấy đặc trưng môi trường
- Gộp điểm dương + âm
- Gọi Open-Meteo lấy thời tiết cho từng (tọa độ, ngày)
- Có cache, có retry, có thanh tiến độ (tqdm)
- Xuất `dataset_with_features.csv` (bảng học máy hoàn chỉnh: đặc trưng + label)

### Bước 4: `train_model.py` — Huấn luyện & đánh giá
- Đọc `dataset_with_features.csv`
- Xử lý giá trị thiếu (điểm API lỗi)
- Chia train/test 80/20 (stratify theo label)
- Train Random Forest; train thêm XGBoost để so sánh
- In: Accuracy, AUC-ROC, Precision, Recall
- Xuất biểu đồ: `roc_curve.png`, `confusion_matrix.png`, `feature_importance.png`
- Lưu mô hình đã train ra file (joblib)

### Bước 5: `make_risk_map.py` — Tạo bản đồ nguy cơ
- Tạo lưới ô đều phủ khu vực nghiên cứu (ví dụ toàn VN hoặc 1 tỉnh, mỗi ô ~0.1 độ)
- Lấy đặc trưng môi trường cho tâm mỗi ô (một ngày đại diện, ví dụ ngày khô nóng nhất)
- Dùng mô hình dự đoán xác suất nguy cơ cho từng ô
- Vẽ bản đồ folium phủ màu theo xác suất (xanh→vàng→đỏ)
- Xuất `risk_map.html`

### Bước main: `run_pipeline.py`
- Chạy tuần tự các bước trên, hoặc cho phép chạy từng bước riêng

---

## LƯU Ý QUAN TRỌNG KHI VIẾT CODE

1. **Rate limit API**: Open-Meteo miễn phí nhưng có giới hạn. Với ~25.000 điểm (dương+âm), cân nhắc:
   - Lấy mẫu nhỏ hơn để demo trước (ví dụ 2000 dương + 2000 âm) rồi mở rộng
   - Hoặc gộp các điểm gần nhau + cùng ngày để giảm số request
   - Bắt buộc có cache ra file để chạy lại không mất công gọi lại

2. **Xử lý cột `type`**: nhãn dương đã lọc sẵn, không cần xử lý thêm.

3. **Cân bằng nhãn**: giữ tỉ lệ dương:âm hợp lý (1:1 hoặc 1:2), dùng stratify khi chia tập.

4. **Giải thích được (interpretability)**: feature_importance.png rất quan trọng cho phần thuyết trình — phải xuất ra và dễ đọc.

5. **Mã sạch, comment tiếng Việt**: đối tượng là học sinh THPT, ưu tiên dễ hiểu hơn là tối ưu.

6. **Không hard-code đường dẫn tuyệt đối**: dùng đường dẫn tương đối hoặc tham số.

---

## CẤU TRÚC THƯ MỤC ĐỀ XUẤT
```
fire-risk-prediction/
├── data/
│   ├── fire_labels_vietnam.csv      # nhãn dương (có sẵn)
│   ├── negative_points.csv          # sinh ra ở bước 2
│   └── dataset_with_features.csv    # sinh ra ở bước 3
├── src/
│   ├── load_labels.py
│   ├── generate_negatives.py
│   ├── fetch_features.py
│   ├── train_model.py
│   ├── make_risk_map.py
│   └── run_pipeline.py
├── outputs/
│   ├── roc_curve.png
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   ├── model.joblib
│   └── risk_map.html
├── requirements.txt
└── README.md
```

---

## TIÊU CHÍ HOÀN THÀNH
- [ ] Chạy `run_pipeline.py` ra được mô hình + các biểu đồ đánh giá + bản đồ HTML
- [ ] AUC-ROC in ra rõ ràng (mục tiêu > 0.75 là tốt cho bài toán này)
- [ ] Bản đồ nguy cơ mở được trên trình duyệt, phủ màu hợp lý
- [ ] Code có comment, chạy lại được nhờ cache, không lỗi khi 1 vài điểm API thất bại
