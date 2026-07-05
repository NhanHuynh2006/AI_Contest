# TỪ ĐIỂN DỮ LIỆU — Dữ liệu cháy NASA FIRMS (MODIS C6.1) Việt Nam 2020–2026

Tài liệu giải thích ý nghĩa từng cột và quy trình làm sạch cho bộ dữ liệu cháy 6.5 năm.

---

## CÁC FILE DỮ LIỆU

| File | Số điểm | Mô tả |
|---|---|---|
| `fire_data_raw_vietnam_2020_2026.csv` | 129.242 | **Dữ liệu gốc** gộp toàn bộ (chưa lọc) — để minh bạch & kiểm chứng |
| `fire_labels_vietnam_2020_2026.csv` | **119.411** | **Dữ liệu đã làm sạch** — dùng làm NHÃN để train mô hình |
| `fire_seasonality_2020_2026.png` | — | Biểu đồ tính mùa vụ (dùng trong báo cáo) |

Dữ liệu gộp từ 7 file NASA (mỗi năm 2020–2026), nguồn MODIS C6.1 trên vệ tinh Terra & Aqua.

---

## GIẢI THÍCH TỪNG CỘT

| Cột | Tên đầy đủ | Ý nghĩa | Đơn vị / Giá trị |
|---|---|---|---|
| `latitude` | Vĩ độ | Vị trí tâm điểm cháy (Bắc–Nam) | độ (VN: 8–24) |
| `longitude` | Kinh độ | Vị trí tâm điểm cháy (Đông–Tây) | độ (VN: 102–110) |
| `brightness` | Nhiệt độ sáng kênh 21 | Độ nóng điểm cháy đo qua kênh hồng ngoại chính. Càng cao càng nóng | Kelvin (K) |
| `scan` | Kích thước pixel (quét) | Bề rộng ô ảnh theo chiều quét vệ tinh | km |
| `track` | Kích thước pixel (bay) | Bề rộng ô ảnh theo chiều bay vệ tinh | km |
| `acq_date` | Ngày quan sát | Ngày vệ tinh chụp được điểm cháy | YYYY-MM-DD |
| `year` | Năm | Năm của điểm cháy (mình thêm để phân tích) | 2020–2026 |
| `month` | Tháng | Tháng của điểm cháy (mình thêm để phân tích mùa vụ) | 1–12 |
| `acq_time` | Giờ quan sát | Giờ chụp (giờ quốc tế UTC), dạng HHMM | vd 400 = 04:00 UTC |
| `satellite` | Vệ tinh | Terra hoặc Aqua (2 vệ tinh mang cảm biến MODIS) | Terra / Aqua |
| `instrument` | Cảm biến | Loại cảm biến — luôn là MODIS | MODIS |
| `confidence` | Độ tin cậy | Mức NASA tin đây là cháy thật. **Quan trọng để lọc nhiễu** | 0–100 (càng cao càng chắc) |
| `version` | Phiên bản xử lý | `61.03`=chất lượng khoa học, `6.1NRT`=near real-time | text |
| `bright_t31` | Nhiệt độ sáng kênh 31 | Nhiệt độ đo qua kênh hồng ngoại thứ 2, để so với nền xung quanh | Kelvin (K) |
| `frp` | Fire Radiative Power | **Công suất bức xạ nhiệt** — cường độ/mức độ dữ dội của điểm cháy | Megawatt (MW) |
| `daynight` | Ngày/Đêm | Chụp ban ngày (D) hay ban đêm (N) | D / N |
| `type` | Loại nguồn nhiệt | Phân loại nguồn cháy (xem bảng dưới) | 0/1/2/3 |
| `source_file` | Nguồn file | Điểm đến từ file archive hay nrt | archive / nrt |
| `label` | Nhãn | Nhãn học máy: 1 = đã từng cháy | 1 |

### Chi tiết cột `type`:
| Giá trị | Ý nghĩa |
|---|---|
| **0** | **Cháy thực vật** (rừng, đồng cỏ, cây trồng) — CÁI TA CẦN |
| 1 | Núi lửa |
| 2 | Nguồn nhiệt tĩnh khác (lò công nghiệp, gas flare) — ĐÃ LOẠI |
| 3 | Cháy ngoài khơi |

---

## QUY TRÌNH LÀM SẠCH (giống hệt cách xử lý dữ liệu 2026)

Mỗi điểm cháy dùng làm **nhãn "nơi này đã từng cháy"**. Để nhãn sạch, đã loại 2 loại nhiễu:

| Bước | Thao tác | Số điểm bỏ |
|---|---|---|
| 0 | Dữ liệu gốc gộp toàn bộ | 129.242 |
| 1 | Loại confidence < 30 (phát hiện không chắc chắn) | −6.556 |
| 2 | Loại type = 2 (nguồn nhiệt công nghiệp, không phải cháy thực vật) | −3.255 |
| 3 | Giữ trong khung Việt Nam | −20 |
| | **Còn lại (nhãn dương sạch)** | **119.411** |

> File gốc vẫn được giữ để đội tự kiểm chứng hoặc thử ngưỡng lọc khác. Trình bày rõ quy trình này là điểm cộng khoa học khi thuyết trình.

---

## PHÂN BỐ DỮ LIỆU

### Theo năm (khá ổn định, không năm nào thiếu bất thường):
| Năm | Số điểm cháy |
|---|---|
| 2020 | 21.157 |
| 2021 | 17.560 |
| 2022 | 11.362 |
| 2023 | 22.021 |
| 2024 | 21.046 |
| 2025 | 13.621 |
| 2026 | 12.644 (đến 01/07) |

### Theo tháng (tính mùa vụ RÕ RỆT — gộp mọi năm):
| Tháng | Số điểm | | Tháng | Số điểm |
|---|---|---|---|---|
| 1 | 12.443 | | 7 | 2.832 |
| 2 | 15.768 | | 8 | 4.101 |
| **3** | **29.286** | | 9 | 2.774 |
| **4** | **29.010** | | 10 | 2.117 |
| 5 | 8.573 | | 11 | 3.104 |
| 6 | 4.667 | | 12 | 4.736 |

**Kết luận khoa học:** Cháy tập trung mạnh vào **tháng 3–4 (đỉnh mùa khô)**, thấp nhất vào tháng 9–10. Đây là bằng chứng thống kê từ 6.5 năm dữ liệu — rất giá trị để đưa vào báo cáo và làm feature "tháng/mùa" cho mô hình.

---

## VÌ SAO DỮ LIỆU NHIỀU NĂM TỐT HƠN CHO MÔ HÌNH
- Học được quy luật mùa vụ thật (dựa 6.5 năm, không suy đoán từ vài tháng)
- Nhãn dương phong phú (119k điểm) → mô hình tổng quát tốt, ít may rủi
- Cho phép phân tích xu hướng qua các năm — điểm cộng nghiên cứu
