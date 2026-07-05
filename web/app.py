# -*- coding: utf-8 -*-
"""
app.py — Web demo trực quan hóa dự báo nguy cơ cháy tại Việt Nam.

Người dùng chọn NGÀY -> web lấy thời tiết ngày đó, chạy mô hình, và tô màu
bản đồ Việt Nam theo mức nguy cơ cháy (xanh -> vàng -> đỏ).

Chạy:
    python web/app.py
Rồi mở trình duyệt: http://127.0.0.1:5000
"""

import sys
import json
from pathlib import Path
from datetime import date, timedelta

# Cho phép import các module trong src/
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from flask import Flask, jsonify, request, render_template
import joblib

from config import MODEL_FILE, RISK_MAP_DATE, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX
from predict_service import predict_grid_for_date
from heatmap import render_heatmap
from plot_3d_risk import plot_3d
import weather_api

# Web bật "thất bại nhanh": nếu Open-Meteo hết lượt, bỏ ngay để chuyển sang phao
# ERA5/GEE (thay vì chờ retry vài phút). Pipeline offline không bật cờ này.
weather_api.set_fast_fail(True)

app = Flask(__name__)

# Nơi lưu các ảnh nhiệt đã dựng (cache theo ngày) để hiện lại tức thì
HEATMAP_DIR = ROOT / "web" / "static" / "heatmaps"
HEATMAP_DIR.mkdir(parents=True, exist_ok=True)
# Nơi lưu ảnh biểu đồ CỘT 3D (cache theo ngày)
PLOT3D_DIR = ROOT / "web" / "static" / "plots3d"
PLOT3D_DIR.mkdir(parents=True, exist_ok=True)

# Giới hạn ngày hợp lệ: quá khứ tới 2000, TƯƠNG LAI tới +15 ngày (dùng dự báo)
MIN_DATE = "2000-01-01"
MAX_DATE = (date.today() + timedelta(days=15)).isoformat()
# Ngày >= mốc này được coi là DỰ BÁO (dùng forecast API thay vì dữ liệu lịch sử)
FORECAST_FROM = (date.today() - timedelta(days=4)).isoformat()


@app.route("/")
def index():
    return render_template(
        "index.html",
        default_date=date.today().isoformat(),   # mở web -> dự báo NGÀY HIỆN TẠI
        min_date=MIN_DATE,
        max_date=MAX_DATE,
        bbox=[LAT_MIN, LAT_MAX, LON_MIN, LON_MAX],
    )


@app.route("/api/info")
def info():
    """Thông tin mô hình + tầm quan trọng đặc trưng (để hiển thị)."""
    bundle = joblib.load(MODEL_FILE)
    model = bundle["model"]
    feats = bundle["feature_columns"]
    imp = getattr(model, "feature_importances_", None)
    labels = {
        "temp_max": "Nhiệt độ tối đa", "humidity_mean": "Độ ẩm TB",
        "wind_max": "Gió tối đa", "precip_sum": "Mưa trong ngày",
        "precip_7d": "Mưa 7 ngày", "precip_14d": "Mưa 14 ngày",
        "dry_days_14d": "Số ngày khô/14", "elevation": "Độ cao",
        "month": "Tháng", "ndvi": "Thảm thực vật (NDVI)",
    }
    importance = []
    if imp is not None:
        pairs = sorted(zip(feats, imp), key=lambda x: -x[1])
        importance = [{"name": labels.get(f, f), "value": round(float(v), 3)}
                      for f, v in pairs]
    return jsonify({
        "model_name": bundle.get("model_name", "?"),
        "features": feats,
        "importance": importance,
    })


def _get_or_make_heatmap(d, step):
    """Lấy ảnh nhiệt của 1 ngày từ cache đĩa, hoặc dựng mới nếu chưa có.
    Trả về dict {date, image, bounds, stats, step, is_forecast}."""
    is_forecast = d >= FORECAST_FROM
    fname = f"{d}_{step}.png"
    fpath = HEATMAP_DIR / fname
    meta_path = HEATMAP_DIR / f"{d}_{step}.json"
    # Cache cả ngày DỰ BÁO: mỗi ngày chỉ gọi API forecast 1 LẦN rồi lưu -> tiết
    # kiệm quota (dự báo không đổi đáng kể trong 1 buổi demo).
    if fpath.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())              # đã dựng -> dùng lại
    else:
        meta = render_heatmap(d, step=step, out_path=str(fpath))
        meta_path.write_text(json.dumps(meta))
    v = int(fpath.stat().st_mtime)                            # chống cache trình duyệt
    return {
        "date": d,
        "image": f"/static/heatmaps/{fname}?v={v}",
        "bounds": meta["bounds"],
        "stats": meta["stats"],
        "grid": meta.get("grid", []),
        "step": step,
        "is_forecast": is_forecast,
    }


@app.route("/api/heatmap")
def heatmap():
    """Dựng bản đồ NHIỆT MƯỢT cho 1 ngày. ?date=YYYY-MM-DD&step=0.5"""
    d = request.args.get("date", RISK_MAP_DATE)
    try:
        step = float(request.args.get("step", 0.5))
    except ValueError:
        step = 0.5
    d = min(max(d, MIN_DATE), MAX_DATE)
    try:
        return jsonify(_get_or_make_heatmap(d, step))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _get_or_make_plot3d(d):
    """Sinh (hoặc lấy từ cache) ảnh biểu đồ CỘT 3D cho 1 ngày.
    Dùng lại LƯỚI đã tính trong bản đồ nhiệt (cache) -> khỏi chạy lại mô hình.
    Có cache theo ngày (kể cả dự báo) để không gọi lại mô hình/API."""
    fname = f"risk3d_{d}.png"
    fpath = PLOT3D_DIR / fname
    if not fpath.exists():
        meta = _get_or_make_heatmap(d, 0.5)      # lấy lưới điểm (có cache)
        plot_3d(d, out_path=str(fpath), points=meta.get("grid", []))
    v = int(fpath.stat().st_mtime)
    return {"date": d, "image": f"/static/plots3d/{fname}?v={v}"}


@app.route("/api/plot3d")
def plot3d():
    """Sinh biểu đồ CỘT 3D nguy cơ cho 1 ngày. ?date=YYYY-MM-DD"""
    d = request.args.get("date", RISK_MAP_DATE)
    d = min(max(d, MIN_DATE), MAX_DATE)
    try:
        return jsonify(_get_or_make_plot3d(d))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Các mốc thời gian cho hoạt ảnh time-lapse (ngày 15 mỗi tháng, mùa cháy 2026)
TIMELAPSE_DATES = [
    ("2026-01-15", "Tháng 1"), ("2026-02-15", "Tháng 2"),
    ("2026-03-15", "Tháng 3"), ("2026-04-15", "Tháng 4"),
    ("2026-05-15", "Tháng 5"), ("2026-06-15", "Tháng 6"),
]


@app.route("/api/timelapse")
def timelapse():
    """Trả về chuỗi khung hình (ảnh nhiệt) theo tháng để chạy hoạt ảnh."""
    step = 0.5
    frames = []
    for d, label in TIMELAPSE_DATES:
        try:
            fr = _get_or_make_heatmap(d, step)
            fr["label"] = label
            frames.append(fr)
        except Exception:
            pass       # ngày nào lỗi (rate-limit) thì bỏ qua, vẫn chạy các ngày khác
    return jsonify({"frames": frames})


@app.route("/api/predict")
def predict():
    """Dự đoán nguy cơ cho 1 ngày. Tham số: ?date=YYYY-MM-DD&step=0.3"""
    d = request.args.get("date", RISK_MAP_DATE)
    try:
        step = float(request.args.get("step", 0.3))
    except ValueError:
        step = 0.3
    # chặn khoảng ngày cho an toàn
    if d < MIN_DATE:
        d = MIN_DATE
    if d > MAX_DATE:
        d = MAX_DATE
    try:
        result = predict_grid_for_date(d, step=step)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("=" * 55)
    print("  WEB DEMO: Dự báo nguy cơ cháy Việt Nam")
    print("  Mở trình duyệt: http://127.0.0.1:5000")
    print(f"  Khoảng ngày hợp lệ: {MIN_DATE} -> {MAX_DATE}")
    print("=" * 55)
    app.run(host="127.0.0.1", port=5000, debug=False)
