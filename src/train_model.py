# -*- coding: utf-8 -*-
"""
Bước 4 — train_model.py: Huấn luyện & đánh giá mô hình dự báo nguy cơ cháy.

- Đọc bảng đặc trưng (dataset_with_features.csv)
- Xử lý giá trị thiếu (điểm API lỗi) bằng cách điền trung vị
- Chia train/test 80/20 (stratify theo label để giữ tỉ lệ dương/âm)
- Train Random Forest (chính) + XGBoost (so sánh)
- In: Accuracy, AUC-ROC, Precision, Recall
- Xuất biểu đồ: roc_curve.png, confusion_matrix.png, feature_importance.png
- Lưu mô hình tốt nhất ra model.joblib (kèm danh sách cột & median để tái sử dụng)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")            # backend không cần màn hình -> lưu PNG là được
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    roc_curve, confusion_matrix,
)
import joblib
from xgboost import XGBClassifier

from config import (
    DATASET_CSV, FEATURE_COLUMNS, RANDOM_SEED,
    MODEL_FILE, ROC_PNG, CM_PNG, FI_PNG, SHAP_PNG,
)

# Nhãn tiếng Việt cho từng đặc trưng (để biểu đồ dễ đọc khi thuyết trình)
FEATURE_LABELS = {
    "temp_max": "Nhiệt độ tối đa",
    "humidity_mean": "Độ ẩm TB",
    "wind_max": "Gió tối đa",
    "precip_sum": "Mưa trong ngày",
    "precip_7d": "Mưa 7 ngày",
    "precip_14d": "Mưa 14 ngày",
    "dry_days_14d": "Số ngày khô/14",
    "elevation": "Độ cao",
    "month": "Tháng",
    "ndvi": "Thảm thực vật (NDVI)",
}


def _load_dataset():
    """Đọc dataset, làm sạch, tách X/y. Trả về thêm cột năm để chia theo thời gian."""
    df = pd.read_csv(DATASET_CSV)

    # Bỏ những dòng thiếu HOÀN TOÀN thời tiết (API hỏng nặng) — hiếm
    df = df.dropna(subset=FEATURE_COLUMNS, how="all").reset_index(drop=True)

    # Loại điểm ÂM rơi xuống biển (độ cao <= 0) — vành đai 7-60km đã hạn chế
    # nhưng vài điểm ven bờ vẫn lọt; giữ lại sẽ làm mô hình học mẹo "biển không cháy"
    n_before = len(df)
    sea_neg = (df["label"] == 0) & (df["elevation"].fillna(-1) <= 0)
    df = df[~sea_neg].reset_index(drop=True)
    n_dropped_sea = n_before - len(df)

    # Đặc trưng động: nếu đã chạy bước NDVI (fetch_ndvi.py) và đủ dữ liệu
    # thì thêm 'ndvi' vào danh sách; nếu không, dùng bộ đặc trưng chuẩn.
    feat_cols = list(FEATURE_COLUMNS)
    if "ndvi" in df.columns and df["ndvi"].notna().mean() >= 0.5:
        feat_cols.append("ndvi")
        # Fix NDVI format issue: GEE trả '[5E-1]' string -> convert to float
        def _fix_ndvi(x):
            if pd.isna(x): return np.nan
            if isinstance(x, str):
                try: return float(x.strip('[]'))
                except: return np.nan
            return float(x)
        df["ndvi"] = df["ndvi"].apply(_fix_ndvi)

    X = df[feat_cols].copy()
    y = df["label"].astype(int).values
    years = pd.to_datetime(df["acq_date"]).dt.year.values

    # Điền giá trị thiếu bằng TRUNG VỊ của từng cột (bền với ngoại lệ)
    medians = X.median(numeric_only=True)
    X = X.fillna(medians)
    return X, y, years, medians, n_dropped_sea, feat_cols


def _evaluate(name, model, X_test, y_test):
    """Tính các chỉ số đánh giá và in ra. Trả về dict kết quả + xác suất dự đoán."""
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "auc": roc_auc_score(y_test, proba),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "proba": proba,
        "pred": pred,
    }
    print(f"  [{name}]")
    print(f"     Accuracy : {metrics['accuracy']:.3f}")
    print(f"     AUC-ROC  : {metrics['auc']:.3f}")
    print(f"     Precision: {metrics['precision']:.3f}")
    print(f"     Recall   : {metrics['recall']:.3f}")
    return metrics


def _plot_roc(y_test, results):
    """Vẽ ROC curve cho cả 2 mô hình chồng lên nhau."""
    plt.figure(figsize=(6, 5))
    for name, r in results.items():
        fpr, tpr, _ = roc_curve(y_test, r["proba"])
        plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={r['auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Đoán ngẫu nhiên")
    plt.xlabel("Tỉ lệ báo động giả (FPR)")
    plt.ylabel("Tỉ lệ phát hiện đúng (TPR)")
    plt.title("Đường cong ROC — dự báo nguy cơ cháy")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(ROC_PNG, dpi=120)
    plt.close()


def _plot_confusion(y_test, best_name, best_result):
    """Vẽ confusion matrix cho mô hình tốt nhất."""
    cm = confusion_matrix(y_test, best_result["pred"])
    plt.figure(figsize=(5, 4.5))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"Ma trận nhầm lẫn — {best_name}")
    ticks = ["Không cháy (0)", "Cháy (1)"]
    plt.xticks([0, 1], ticks)
    plt.yticks([0, 1], ticks)
    plt.xlabel("Mô hình dự đoán")
    plt.ylabel("Thực tế")
    # In số lượng lên từng ô
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontsize=13, fontweight="bold")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(CM_PNG, dpi=120)
    plt.close()


def _plot_shap(model, X_test):
    """
    Biểu đồ SHAP — chuẩn 'giải thích mô hình' trong các bài báo cháy rừng 2024–2025.

    Khác feature_importance thường (chỉ nói đặc trưng nào QUAN TRỌNG), SHAP nói cả
    CHIỀU tác động: ví dụ 'mưa 14 ngày THẤP đẩy nguy cơ LÊN, độ ẩm CAO kéo nguy cơ
    XUỐNG'. Rất mạnh khi thuyết trình và phản biện.
    """
    try:
        import shap
        sample = X_test.sample(n=min(800, len(X_test)), random_state=RANDOM_SEED)
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(sample)
        # RF trả list [lớp 0, lớp 1] hoặc mảng 3 chiều; lấy phần của lớp 1 (cháy)
        if isinstance(sv, list):
            sv = sv[1]
        elif getattr(sv, "ndim", 2) == 3:
            sv = sv[:, :, 1]
        labels = [FEATURE_LABELS.get(c, c) for c in sample.columns]
        plt.figure()
        shap.summary_plot(sv, sample, feature_names=labels, show=False)
        plt.title("SHAP — đặc trưng đẩy nguy cơ cháy lên/xuống thế nào")
        plt.tight_layout()
        plt.savefig(SHAP_PNG, dpi=120, bbox_inches="tight")
        plt.close("all")
        return True
    except Exception as e:
        print(f"  (Bỏ qua SHAP: {e})")
        return False


def _plot_feature_importance(model, feature_cols):
    """Vẽ tầm quan trọng đặc trưng (rất quan trọng cho phần thuyết trình)."""
    importances = model.feature_importances_
    order = np.argsort(importances)          # tăng dần để bar dài nằm trên
    labels = [FEATURE_LABELS.get(feature_cols[i], feature_cols[i]) for i in order]
    vals = importances[order]

    plt.figure(figsize=(7, 4.5))
    plt.barh(labels, vals, color="#d95f02")
    plt.xlabel("Mức độ quan trọng")
    plt.title("Đặc trưng nào quyết định nguy cơ cháy? (Random Forest)")
    for y, v in enumerate(vals):
        plt.text(v, y, f" {v:.3f}", va="center")
    plt.tight_layout()
    plt.savefig(FI_PNG, dpi=120)
    plt.close()


def train_model(verbose=True):
    """Huấn luyện, đánh giá, vẽ biểu đồ, lưu mô hình. Trả về dict tóm tắt."""
    if verbose:
        print("=" * 60)
        print("BƯỚC 4: HUẤN LUYỆN & ĐÁNH GIÁ MÔ HÌNH")
        print("=" * 60)

    X, y, years, medians, n_dropped_sea, feat_cols = _load_dataset()
    if verbose:
        print(f"  Dữ liệu: {len(X):,} điểm, {len(feat_cols)} đặc trưng")
        print(f"  (Đã loại {n_dropped_sea} điểm âm rơi xuống biển)")
        print(f"  Đặc trưng: {', '.join(feat_cols)}")
        print(f"  Phân bố nhãn: dương={int((y==1).sum()):,}, âm={int((y==0).sum()):,}")
        print(f"  Khoảng năm: {years.min()}–{years.max()}")
        print()

    # Chia 80/20, stratify để giữ tỉ lệ dương/âm ở cả 2 tập
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
    )
    if verbose:
        print(f"  Tập train: {len(X_train):,}  |  Tập test: {len(X_test):,}")
        print()

    # Lấy 2 mô hình NGUYÊN MẪU (chưa train) từ model_prototype.py rồi HUẤN LUYỆN.
    spw = float((y_train == 0).sum()) / max(1, (y_train == 1).sum())
    from model_prototype import build_random_forest, build_xgboost

    # --- Mô hình 1: Random Forest (chính) ---
    rf = build_random_forest()
    rf.fit(X_train, y_train)

    # --- Mô hình 2: XGBoost (so sánh) ---
    xgb = build_xgboost(scale_pos_weight=spw)
    xgb.fit(X_train, y_train)

    if verbose:
        print("  KẾT QUẢ ĐÁNH GIÁ (chia ngẫu nhiên 80/20):")
    results = {
        "Random Forest": _evaluate("Random Forest", rf, X_test, y_test),
        "XGBoost": _evaluate("XGBoost", xgb, X_test, y_test),
    }

    # Chọn mô hình tốt nhất theo AUC
    best_name = max(results, key=lambda k: results[k]["auc"])
    best_model = rf if best_name == "Random Forest" else xgb
    if verbose:
        print(f"\n  => Mô hình tốt nhất theo AUC: {best_name} "
              f"(AUC={results[best_name]['auc']:.3f})")

    # --- KIỂM ĐỊNH THEO THỜI GIAN (nghiêm ngặt & thuyết phục nhất) ---
    # Train trên 2020–2024, test trên 2025–2026: mô phỏng đúng tình huống thật
    # "dùng quá khứ dự báo tương lai". AUC ở đây là con số đáng tin để báo cáo.
    tr_mask = years <= 2024
    te_mask = years >= 2025
    if tr_mask.sum() > 500 and te_mask.sum() > 200 and len(set(y[te_mask])) == 2:
        if verbose:
            print()
            print("  KIỂM ĐỊNH THEO THỜI GIAN (train 2020–2024 → test 2025–2026):")
            print(f"     Train: {int(tr_mask.sum()):,} điểm | Test: {int(te_mask.sum()):,} điểm")
        rf_t = build_random_forest().fit(X[tr_mask], y[tr_mask])
        xgb_t = build_xgboost().fit(X[tr_mask], y[tr_mask])
        _evaluate("RF  (temporal)", rf_t, X[te_mask], y[te_mask])
        _evaluate("XGB (temporal)", xgb_t, X[te_mask], y[te_mask])

    # --- Vẽ biểu đồ ---
    _plot_roc(y_test, results)
    _plot_confusion(y_test, best_name, results[best_name])
    # Feature importance: dùng Random Forest cho dễ giải thích
    _plot_feature_importance(rf, feat_cols)
    # SHAP: giải thích chiều tác động của từng đặc trưng.
    # Dùng Random Forest (sklearn) — TreeExplainer đọc cây sklearn ổn định hơn
    # XGBoost (bản XGBoost mới format ngưỡng cây theo kiểu SHAP chưa parse được).
    _plot_shap(rf, X_test)

    # --- Lưu mô hình + thông tin cần để dự đoán lại sau này ---
    joblib.dump(
        {
            "model": best_model,
            "model_name": best_name,
            "feature_columns": feat_cols,
            "medians": medians,      # để điền giá trị thiếu khi dự đoán bản đồ
        },
        MODEL_FILE,
    )

    if verbose:
        print()
        print(f"  Đã lưu biểu đồ: {ROC_PNG.name}, {CM_PNG.name}, {FI_PNG.name}")
        print(f"  Đã lưu mô hình: {MODEL_FILE.name}")
        print()

    return {
        "best_name": best_name,
        "results": {k: {m: v[m] for m in ["accuracy", "auc", "precision", "recall"]}
                    for k, v in results.items()},
    }


if __name__ == "__main__":
    train_model()
