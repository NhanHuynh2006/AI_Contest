# -*- coding: utf-8 -*-
"""Chạy nền: chờ quota Open-Meteo mở rồi tái tạo 6 frame với mưa 14-ngày THẬT."""
import sys, time, os, subprocess
sys.path.insert(0, "src")
os.chdir("/home/nolan/Documents/AI_contest")

PROBE = ("https://archive-api.open-meteo.com/v1/archive?latitude=15&longitude=108"
         "&start_date=2026-04-01&end_date=2026-04-14&daily=temperature_2m_max")


def quota_open():
    import requests
    try:
        return requests.get(PROBE, timeout=15).status_code == 200
    except Exception:
        return False


# 1) Chờ quota
while not quota_open():
    print(time.strftime("%H:%M"), "probe -> chưa mở, chờ 5 phút", flush=True)
    time.sleep(300)
print("=== QUOTA MỞ — xoá grid median & fetch mưa thật ===", flush=True)

# 2) Xoá các dòng grid median-fill khỏi v2 cache để buộc fetch lại đúng
import pandas as pd
from predict_service import _build_grid
dates = ["2026-01-15", "2026-02-15", "2026-03-15",
         "2026-04-15", "2026-05-15", "2026-06-15"]
need = set()
for d in dates:
    for r in _build_grid(d, step=0.5).itertuples(index=False):
        need.add(f"{r.latitude}_{r.longitude}_{d}")
v2 = pd.read_csv("data/cache/weather_cache_v2.csv")
v2 = v2[~v2["key"].astype(str).isin(need)]
v2.to_csv("data/cache/weather_cache_v2.csv", index=False)
print(f"Đã xoá {len(need)} dòng grid median -> sẽ fetch mưa 14-ngày thật", flush=True)

# 3) Tái tạo 6 frame — ghi CẢ png lẫn json (đúng định dạng app.py đọc)
import json
from heatmap import render_heatmap
os.makedirs("web/static/heatmaps", exist_ok=True)
for i, d in enumerate(dates, 1):
    t0 = time.time()
    png = f"web/static/heatmaps/{d}_0.5.png"
    meta = render_heatmap(d, step=0.5, out_path=png)
    with open(f"web/static/heatmaps/{d}_0.5.json", "w") as f:
        json.dump(meta, f)
    s = meta["stats"]
    print(f"[{i}/6] {d}  mean={s['mean_risk']} max={s['max_risk']} "
          f"high={s['n_high']}  ({time.time()-t0:.0f}s)", flush=True)
print("=== ĐÊM: TÁI TẠO 6 FRAME (MƯA THẬT) XONG ===", flush=True)
