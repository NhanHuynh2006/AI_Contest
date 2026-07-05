# TỪ ĐIỂN DỮ LIỆU — Dữ liệu cháy NASA FIRMS (MODIS C6.1) Việt Nam

Tài liệu này giải thích ý nghĩa từng cột trong dữ liệu điểm cháy, và lý do vì sao có bước làm sạch.

---

## CÓ 2 FILE DỮ LIỆU

| File | Số điểm | Mô tả |
|---|---|---|
| `fire_data_raw_vietnam.csv` | 13.252 | **Dữ liệu gốc** gộp từ NASA (chưa lọc gì) — để tham khảo & minh bạch |
| `fire_labels_vietnam.csv` | 12.644 | **Dữ liệu đã làm sạch** — dùng làm NHÃN để train mô hình |

Cả 2 đều gộp từ 2 file NASA gửi: `archive` (chất lượng khoa học, tháng 1–2) và `nrt` (near real-time, tháng 3–7).

---

## GIẢI THÍCH TỪNG CỘT

| Cột | Tên đầy đủ | Ý nghĩa | Đơn vị / Giá trị |
|---|---|---|---|
| `latitude` | Vĩ độ | Vị trí tâm điểm cháy (Bắc–Nam) | độ (VN: 8–24) |
| `longitude` | Kinh độ | Vị trí tâm điểm cháy (Đông–Tây) | độ (VN: 102–110) |
| `brightness` | Nhiệt độ sáng kênh 21 | Độ nóng của điểm cháy đo qua kênh hồng ngoại chính. Càng cao càng nóng | Kelvin (K) |
| `scan` | Kích thước pixel (quét) | Bề rộng ô ảnh theo chiều quét của vệ tinh | km |
| `track` | Kích thước pixel (bay) | Bề rộng ô ảnh theo chiều bay của vệ tinh | km |
| `acq_date` | Ngày quan sát | Ngày vệ tinh chụp được điểm cháy | YYYY-MM-DD |
| `acq_time` | Giờ quan sát | Giờ chụp (giờ quốc tế UTC), dạng HHMM | vd 747 = 07:47 UTC |
| `satellite` | Vệ tinh | Terra hoặc Aqua (2 vệ tinh mang cảm biến MODIS) | Terra / Aqua |
| `instrument` | Cảm biến | Loại cảm biến — ở đây luôn là MODIS | MODIS |
| `confidence` | Độ tin cậy | Mức độ NASA tin đây là cháy thật. **Rất quan trọng để lọc nhiễu** | 0–100 (càng cao càng chắc) |
| `version` | Phiên bản xử lý | Phiên bản thuật toán. `61.03`=chất lượng khoa học, `6.1NRT`=near real-time | text |
| `bright_t31` | Nhiệt độ sáng kênh 31 | Nhiệt độ đo qua kênh hồng ngoại thứ 2, dùng để so sánh với nền xung quanh | Kelvin (K) |
| `frp` | Fire Radiative Power | **Công suất bức xạ nhiệt** — thể hiện cường độ/mức độ dữ dội của điểm cháy. Càng cao đám cháy càng mạnh | Megawatt (MW) |
| `daynight` | Ngày/Đêm | Điểm cháy chụp ban ngày (D) hay ban đêm (N) | D / N |
| `type` | Loại nguồn nhiệt | **Phân loại nguồn cháy** (chỉ file archive có). Xem bảng bên dưới | 0/1/2/3 |
| `source_file` | Nguồn file | Điểm này đến từ file archive hay nrt (do mình thêm để phân biệt) | archive / nrt |
| `month` | Tháng | Tháng của điểm cháy (do mình thêm để phân tích theo mùa) | 1–12 |
| `label` | Nhãn | Nhãn học máy: 1 = đã từng cháy (do mình thêm) | 1 |

### Chi tiết cột `type` (loại nguồn nhiệt):
| Giá trị | Ý nghĩa |
|---|---|
| **0** | **Cháy thực vật** (rừng, đồng cỏ, cây trồng) — ĐÂY LÀ CÁI TA CẦN |
| 1 | Núi lửa |
| 2 | Nguồn nhiệt tĩnh khác (lò công nghiệp, gas flare, nhà máy) |
| 3 | Cháy ngoài khơi (offshore) |

---

## VÌ SAO CẦN LÀM SẠCH (không dùng thẳng file gốc)

Trong đề tài này, mỗi điểm cháy được dùng làm **nhãn "nơi này đã từng cháy"** để dạy mô hình. Nếu nhãn sai, mô hình học sai theo. Vì vậy đã loại 2 loại điểm gây nhiễu:

**1. Loại điểm confidence < 30 (bỏ 577 điểm)**
Đây là những phát hiện mà NASA cũng không chắc là cháy thật — có thể là mái tôn nóng, mây mỏng, hoặc nhiễu cảm biến. Dùng làm nhãn "đã cháy" sẽ dạy mô hình sai.

**2. Loại điểm type = 2 (bỏ 31 điểm)**
Đây là nguồn nhiệt công nghiệp/gas flare — nóng thật nhưng KHÔNG PHẢI cháy thực vật. Đề tài dự báo nguy cơ *cháy*, nên loại các nguồn nhiệt nhân tạo này để nhãn đúng bản chất.

**Kết quả:** từ 13.252 điểm gốc → còn **12.644 điểm cháy thực vật đáng tin**, dùng làm nhãn dương sạch.

> **Ghi chú minh bạch:** File gốc vẫn được giữ (`fire_data_raw_vietnam.csv`) để đội có thể tự kiểm chứng, thử ngưỡng lọc khác, hoặc giải thích cho ban giám khảo cách dữ liệu được xử lý. Việc trình bày rõ quy trình làm sạch này là một điểm cộng khoa học khi thuyết trình.

---

## PHÂN BỐ THEO THÁNG (dữ liệu đã sạch)

| Tháng | Số điểm cháy | Ghi chú |
|---|---|---|
| 1 | 1.823 | |
| 2 | 1.234 | |
| 3 | 2.561 | |
| **4** | **5.053** | **Cao nhất — đỉnh mùa khô** |
| 5 | 1.157 | |
| 6 | 782 | Vào mùa mưa, cháy giảm |
| 7 | 34 | (chỉ có 1 ngày dữ liệu) |

Xu hướng này rất hợp lý về mặt khoa học: cháy tập trung vào mùa khô (tháng 3–4), giảm mạnh khi vào mùa mưa — một điểm hay để phân tích và trình bày trong báo cáo.
