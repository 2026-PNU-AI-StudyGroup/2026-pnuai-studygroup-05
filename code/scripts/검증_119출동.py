# -*- coding: utf-8 -*-
"""축5 검증: 부산 119 소방출동정보 API로 동구 구급 출동을 동별 집계 → 우리 위험도와 대조"""
import sys, time, re
sys.stdout.reconfigure(encoding="utf-8")

# ── 저장소 루트로 이동: 실행 위치와 무관하게 data/·outputs/ 상대경로 유지 ──
import os
from pathlib import Path
for _c in [Path.cwd(), *Path.cwd().parents]:
    if (_c / "README.md").exists() and (_c / ".gitignore").exists():
        os.chdir(_c)
        break

import requests, pandas as pd, geopandas as gpd
import xml.etree.ElementTree as ET

URL  = "https://apis.data.go.kr/6260000/Busan119InfoService/getTodayInfo"
NUM  = 1000            # 페이지당 레코드
N_PAGES = 40           # 최근 N페이지(≈4만건) 표본

# --- 인증키(.env) ---
key = None
for line in open(".env", encoding="utf-8"):
    if line.startswith("DATA_GO_KR_SERVICE_KEY"):
        key = line.split("=", 1)[1].strip()

def fetch(page):
    for _ in range(3):
        try:
            r = requests.get(URL, params={"serviceKey": key, "numOfRows": NUM, "pageNo": page}, timeout=30)
            root = ET.fromstring(r.content)
            total = root.findtext(".//totalCount")
            rows = [{c.tag: c.text for c in it} for it in root.findall(".//item")]
            return rows, (int(total) if total else None)
        except Exception:
            time.sleep(2)
    return [], None

# --- 총건수 → 최근 페이지 산정 ---
_, total = fetch(1)
last = total // NUM + 1
print(f"부산 119 누적 출동: {total:,}건 | 마지막 페이지 {last}")

# --- 최근 N페이지 수집(최신순) ---
recs = []
for p in range(last, max(1, last - N_PAGES), -1):
    rows, _ = fetch(p)
    recs += rows
    time.sleep(0.15)
df = pd.DataFrame(recs)
df["regtime"] = df["regtime"].astype(str)
print(f"수집: {len(df):,}건 | 기간 {df['regtime'].min()[:8]} ~ {df['regtime'].max()[:8]}")

# --- 동구 구급 필터 + 동 추출 ---
def get_dong(addr):
    a = str(addr).replace(" ", "")
    m = re.search(r"동구(\S+?동)", a)
    return m.group(1) if m else None

dong_gu = df[df["dsraddr"].astype(str).str.contains("동구", na=False)].copy()
dong_gu["dong"] = dong_gu["dsraddr"].apply(get_dong)
gugeup = dong_gu[dong_gu["dsrkndcd"] == "구급"].copy()             # 구급 출동
disease = gugeup[gugeup["dsrclscd"] == "질병"].copy()             # 질병(심정지 계열 근접)
print(f"동구 전체 {len(dong_gu):,} | 구급 {len(gugeup):,} | 구급·질병 {len(disease):,}")

target = ["초량1동", "초량2동", "초량3동", "초량6동", "좌천동"]
c_all  = gugeup [gugeup ["dong"].isin(target)].groupby("dong").size().reindex(target).fillna(0).astype(int)
c_dis  = disease[disease["dong"].isin(target)].groupby("dong").size().reindex(target).fillna(0).astype(int)

# --- 우리 위험도·인구와 대조 ---
oa   = gpd.read_parquet("outputs/oa_risk.parquet")
risk = oa.groupby("dong")["risk_norm"].mean().reindex(target)
p75  = oa.groupby("dong")["p75_2026"].sum().reindex(target)

comp = pd.DataFrame({
    "구급출동": c_all, "구급·질병": c_dis,
    "위험도(우리)": risk.round(3), "75+인구": p75.round(0).astype(int),
})
print("\n[대상 5동 — 실제 119 구급 출동 vs 우리 추정]")
print(comp.to_string())

print("\n[상관계수] (표본이라 참고용)")
print(f"  구급출동 ↔ 위험도 : {comp['구급출동'].corr(comp['위험도(우리)']):.3f}")
print(f"  구급출동 ↔ 75+인구: {comp['구급출동'].corr(comp['75+인구']):.3f}")
print(f"  구급·질병 ↔ 위험도 : {comp['구급·질병'].corr(comp['위험도(우리)']):.3f}")

# --- 저장 ---
os.makedirs("pusan_data", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
gugeup.to_csv("pusan_data/119_동구_구급출동_표본.csv", index=False, encoding="utf-8-sig")
comp.to_csv("outputs/ax5_119_검증.csv", encoding="utf-8-sig")
print("\n저장: pusan_data/119_동구_구급출동_표본.csv · outputs/ax5_119_검증.csv")
