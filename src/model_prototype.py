# -*- coding: utf-8 -*-
"""
model_prototype.py — ĐỊNH NGHĨA MÔ HÌNH NGUYÊN MẪU (chưa huấn luyện).

File này CHỈ mô tả "kiến trúc" mô hình (loại thuật toán + siêu tham số), TRẢ VỀ
các mô hình RỖNG — chưa học gì từ dữ liệu. Đây là "bản thiết kế" của mô hình.

Quy trình 3 file rõ ràng:
    1. model_prototype.py  (file này) — mô hình NGUYÊN MẪU, CHƯA train.
    2. train_model.py                  — nạp dữ liệu, GỌI file này, HUẤN LUYỆN & lưu.
    3. outputs/model.joblib            — FILE WEIGHT: mô hình ĐÃ train xong.

Cách xem thử mô hình nguyên mẫu:
    python src/model_prototype.py
"""

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

RANDOM_SEED = 42


def build_random_forest():
    """Trả về Random Forest NGUYÊN MẪU (chưa .fit) — 400 cây."""
    return RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced_subsample",   # cân bằng lớp cháy/không cháy
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )


def build_xgboost(scale_pos_weight=1.0):
    """
    Trả về XGBoost NGUYÊN MẪU (chưa .fit) — 600 cây, độ sâu 7.
    scale_pos_weight: tỉ lệ cân bằng lớp (tính từ dữ liệu lúc train).
    """
    return XGBClassifier(
        n_estimators=600,
        max_depth=7,
        learning_rate=0.06,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )


def build_models(scale_pos_weight=1.0):
    """Trả về cả 2 mô hình nguyên mẫu để train & so sánh."""
    return {
        "Random Forest": build_random_forest(),
        "XGBoost": build_xgboost(scale_pos_weight=scale_pos_weight),
    }


if __name__ == "__main__":
    print("MÔ HÌNH NGUYÊN MẪU (chưa huấn luyện):\n")
    for name, m in build_models().items():
        print(f"  === {name} ===")
        print("   ", m)
        print()
    print("-> Đây mới là 'bản thiết kế'. Chạy 'python src/train_model.py' để HUẤN LUYỆN.")
