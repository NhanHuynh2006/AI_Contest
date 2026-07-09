# 🧠 MÔ TẢ MÔ HÌNH — Dự báo nguy cơ cháy rừng Việt Nam

*(Model Card — tài liệu mô tả mô hình học máy của dự án)*

---

## 1. Tóm tắt

| Mục | Nội dung |
|---|---|
| **Tên** | VN FireWatch — Mô hình dự báo nguy cơ cháy rừng |
| **Bài toán** | Fire Susceptibility / Risk Prediction (dự báo *nguy cơ* một nơi có thể cháy) |
| **Loại** | Phân loại nhị phân (cháy / không cháy) → xuất **xác suất nguy cơ 0–100%** |
| **Thuật toán** | **XGBoost** (Extreme Gradient Boosting) — chọn qua so sánh với Random Forest |
| **Đầu vào** | 10 đặc trưng môi trường (thời tiết, độ khô, thảm thực vật, địa hình) |
| **Đầu ra** | Xác suất nguy cơ cháy cho từng ô lưới trên lãnh thổ Việt Nam |

> ⚠️ Đây là mô hình dự báo **NGUY CƠ** (nơi nào *dễ* cháy theo điều kiện môi trường),
> **KHÔNG** phải phát hiện đám cháy đang diễn ra.

---

## 2. Thuật toán: vì sao chọn XGBoost?

**XGBoost** là thuật toán **cây quyết định tăng cường độ dốc** (gradient boosting):
xây dựng lần lượt nhiều cây, mỗi cây mới sửa lỗi của các cây trước → mô hình mạnh dần.

- **Cấu hình:** 600 cây, độ sâu tối đa 7, learning rate 0.06, `scale_pos_weight` cân
  bằng lớp.
- **Quy trình chọn model:** huấn luyện SONG SONG cả **Random Forest** và **XGBoost**,
  rồi tự động chọn mô hình có **AUC cao nhất** trên tập kiểm tra → XGBoost thắng.
- **Vì sao không dùng Deep Learning:** với dữ liệu **dạng bảng** (mỗi điểm = vài con
  số), XGBoost/Random Forest cho độ chính xác cao hơn và nhanh hơn mạng nơ-ron —
  điều đã được nhiều bài báo cháy rừng 2024–2025 (MDPI, Nature Sci. Reports) xác nhận.
  Deep learning chỉ vượt trội với ảnh vệ tinh không–thời gian, cần GPU và dữ liệu lớn.

---

## 3. Dữ liệu huấn luyện

| | Chi tiết |
|---|---|
| **Nhãn dương (đã cháy)** | **119.411 điểm cháy** NASA FIRMS (MODIS C6.1), 2020–2026, 6.5 năm |
| **Nhãn âm (không cháy)** | Sinh bằng *pseudo-absence*: điểm trong **vành đai 7–60 km** quanh điểm cháy (đủ xa để chắc "không cháy", đủ gần để ở trên đất liền) + **trùng phân phối ngày** với điểm dương |
| **Lấy mẫu huấn luyện** | 3.000 điểm/lớp (do giới hạn tốc độ lấy đặc trưng) |
| **Chia dữ liệu** | 80% train / 20% test, stratify theo nhãn |

**Vì sao xử lý nhãn âm kỹ:** nếu điểm âm rơi xuống biển hoặc lệch mùa, mô hình sẽ
"ăn gian" (đoán theo biển/mùa thay vì học điều kiện cháy thật). Vành đai 7–60 km +
trùng ngày buộc mô hình học **quan hệ môi trường → cháy** thực chất.

---

## 4. Đặc trưng đầu vào (10)

| Đặc trưng | Ý nghĩa | Tầm quan trọng |
|---|---|---|
| Độ cao | Địa hình (m) | **0.131** |
| Độ ẩm TB | Độ ẩm không khí (%) | **0.124** |
| Tháng | Nắm bắt tính mùa vụ | 0.110 |
| Nhiệt độ tối đa | °C trong ngày | 0.105 |
| NDVI | Thảm thực vật (nhiên liệu) | 0.102 |
| Mưa trong ngày | mm | 0.099 |
| Số ngày khô/14 | Chuỗi ngày hạn | 0.085 |
| Mưa 7 ngày | Độ ẩm nhiên liệu ngắn hạn | 0.083 |
| Gió tối đa | km/h | 0.082 |
| Mưa 14 ngày | Độ khô tích lũy | 0.079 |

Nhóm **"độ khô tích lũy"** (mưa 7/14 ngày, số ngày khô) và **NDVI** được thêm theo
đúng khuyến nghị các bài báo quốc tế: chỉ số ẩm nhiên liệu (FFMC/DC) và NDVI là các
yếu tố dự báo cháy hàng đầu.

---

## 5. Kết quả đánh giá

| Cách đánh giá | XGBoost | Ý nghĩa |
|---|---|---|
| **Chia ngẫu nhiên 80/20** | **AUC ≈ 0.80** | Độ phân biệt cháy/không cháy tốt |
| **Kiểm định theo thời gian** (train 2020–2024 → test **2025–2026**) | **AUC ≈ 0.76** | Con số ĐÁNG TIN NHẤT: mô phỏng "dùng quá khứ dự báo tương lai" |

> AUC thấp hơn con số 0.9x của bản nháp đầu là **có chủ đích và trung thực hơn** —
> vì nhãn âm giờ khó hơn (vành đai + trùng ngày), mô hình không còn "ăn gian".

**Biểu đồ trong `outputs/`:** `roc_curve.png`, `confusion_matrix.png`,
`feature_importance.png`, `shap_summary.png` (SHAP giải thích chiều tác động từng
đặc trưng — chuẩn "giải thích được" của các nghiên cứu 2024–2025).

---

## 6. Nguồn dữ liệu (đều chuẩn quốc tế)

| Dữ liệu | Nguồn |
|---|---|
| Điểm cháy (nhãn) | **NASA FIRMS** — MODIS C6.1 |
| Thời tiết quá khứ | **ERA5-Land** (Copernicus/ECMWF) qua Google Earth Engine |
| Dự báo thời tiết | **GFS** (NOAA) qua Open-Meteo |
| Thảm thực vật NDVI | **MODIS** MOD13Q1 qua Google Earth Engine |
| Ranh giới lãnh thổ | **geoBoundaries** (ĐH William & Mary) + FAO |

---

## 7. Ứng dụng & giới hạn

**Ứng dụng:**
- 🔮 **Cảnh báo sớm**: dự báo nguy cơ cháy **15 ngày tới** (dùng dự báo thời tiết) →
  giúp kiểm lâm chủ động bố trí lực lượng.
- 🗺️ **Bản đồ nguy cơ theo vùng**: biết ĐÂU nguy hiểm nhất để ưu tiên phòng chống.
- 📊 **Phân tích lịch sử**: vùng/mùa nào rủi ro (đỉnh cháy **tháng 3–4**).

**Giới hạn (trung thực):**
- Dự báo tương lai tối đa **~15–16 ngày** (giới hạn vật lý của dự báo thời tiết toàn
  cầu, không phải giới hạn mô hình). Càng xa càng kém chính xác.
- Mô hình học từ điểm cháy **trong lãnh thổ Việt Nam** → dự đoán ngoài VN chỉ tham khảo.
- Là dự báo **nguy cơ**, không khẳng định chắc chắn sẽ cháy.

---

## 8. Cách mô hình "dự báo tương lai"

Mô hình học **QUAN HỆ**: *nóng + khô + thảm thực vật khô + địa hình → dễ cháy*. Quan
hệ này áp dụng cho **bất kỳ điều kiện thời tiết nào**. Vì vậy:
- Đưa **thời tiết quá khứ** (ERA5) → phân tích nguy cơ đã qua.
- Đưa **dự báo thời tiết** (GFS/Open-Meteo) → **dự báo nguy cơ cháy các ngày tới**.

Cùng một mô hình — chỉ thay nguồn thời tiết đầu vào.
