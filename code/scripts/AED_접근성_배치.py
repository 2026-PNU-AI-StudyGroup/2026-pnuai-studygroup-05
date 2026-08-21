# -*- coding: utf-8 -*-
"""
AED 신규 배치 — 실제 설치 가능 지점(POI) 기반 · 야간 접근성 · 경사 보정 커버리지

기존 MCLP 는 '집계구 중심' 좌표를 냈다(실제 설치 불가능한 지점). 이 스크립트는
실제로 AED 를 둘 수 있는 장소(24시간 편의점·지구대·119안전센터 등)를 후보로 삼고,
산복도로의 경사를 반영한 도보 접근시간으로 커버 여부를 판정한다.

고려 요소
  1) 접근성  — 야간에 실제로 들어갈 수 있는가 (24시간 체인/상주기관 = A등급)
  2) 출동지연 — 구급차가 환자에게 닿는 시간(t1+t2)이 길수록 수요 가중 ↑
  3) 기존 AED — 이미 커버되는 곳에 중복 배치하지 않음 (aed_donggu.csv 있을 때)

실행:  python code/scripts/AED_접근성_배치.py
"""
import os, sys, json, math
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")

for _c in [Path.cwd(), *Path.cwd().parents]:
    if (_c / "README.md").exists() and (_c / ".gitignore").exists():
        os.chdir(_c); break

import numpy as np, pandas as pd

# ── 파라미터 ────────────────────────────────────────────────────────────────
T_ROUND_MIN = 4.0     # 목격자가 AED 왕복에 쓸 수 있는 시간(분). R=150m 평지 ≈ 3.6분
DETOUR      = 1.3     # 직선 → 실제 보행경로 우회율(격자형 통행 통상값)
N_NEW       = 36      # 수요 80% 달성 규모(커버곡선 기준). 상위 N개만 잘라 쓰면 됨
GRADE_W     = {"A": 1.00, "B": 0.55, "C": 0.20}   # 야간 접근 가능성 가중

POI   = "/tmp/aed/poi_elev.csv"
DEM   = "/tmp/aed/demand_elev.csv"
AED   = "outputs/aed_donggu.csv"          # 있으면 기존 커버 차감
OUT   = "data/output/AED_신규배치_접근성.csv"

# ── Tobler 보행함수 (경사 반영) ─────────────────────────────────────────────
def walk_speed(slope):
    """Tobler: 경사 slope(=dz/dx)에서의 보행속도 m/s"""
    return 6.0 * math.exp(-3.5 * abs(slope + 0.05)) * 1000 / 3600

def round_trip_min(dist_m, dz_m):
    """AED 지점까지 갔다 오는 왕복 보행시간(분). 오르막/내리막 각각 계산."""
    d = max(dist_m * DETOUR, 1.0)
    s = dz_m / d
    return (d / walk_speed(s) + d / walk_speed(-s)) / 60.0

def meters(lon1, lat1, lon2, lat2):
    """대상지 위도(35.12°)에서의 평면 근사 거리(m)"""
    return np.hypot((lon1 - lon2) * 91_000, (lat1 - lat2) * 111_000)

# ── 데이터 ──────────────────────────────────────────────────────────────────
poi = pd.read_csv(POI)
dem = pd.read_csv(DEM, dtype={"집계구코드": str})

# 수요 가중치 = 정규화(75세+) × 정규화(구급차 도달시간 t1+t2)
dem["구급차도달_분"] = dem["t1_출동_분"] + dem["t2_접근_들것_분"]
nz = lambda s: (s - s.min()) / (s.max() - s.min())
dem["수요가중"] = nz(dem["75세이상인구_2026"]) * nz(dem["구급차도달_분"])

# 기존 AED 커버 차감(데이터 있을 때만)
covered = np.zeros(len(dem), dtype=bool)
if os.path.exists(AED):
    a = pd.read_csv(AED)
    night = a[a.get("야간접근", True) == True] if "야간접근" in a else a
    for i, r in dem.iterrows():
        d = meters(r.lon, r.lat, night["wgs84Lon"].values, night["wgs84Lat"].values)
        covered[i] = (d <= 150).any()
    print(f"[기존 AED] {len(night)}곳 반영 → 이미 커버 {covered.sum()}/{len(dem)} 집계구 제외")
else:
    print(f"[기존 AED] {AED} 없음 → 기존 커버 차감 생략(잠정 결과)")

dem["잔여수요"] = np.where(covered, 0.0, dem["수요가중"])

# ── 커버 행렬: 후보 POI × 집계구 (경사 반영 왕복시간 ≤ T) ────────────────────
cov = np.zeros((len(poi), len(dem)), dtype=bool)
for i, p in poi.iterrows():
    dist = meters(p.lon, p.lat, dem.lon.values, dem.lat.values)
    for j in range(len(dem)):
        if dist[j] > 600:            # 명백히 먼 조합은 건너뜀(속도)
            continue
        cov[i, j] = round_trip_min(dist[j], dem.elev.values[j] - p.elev) <= T_ROUND_MIN

print(f"[커버] 후보 {len(poi)} × 집계구 {len(dem)} — 커버 쌍 {cov.sum()}개")

# ── 3단계 배치 전략 ────────────────────────────────────────────────────────
#   Tier1  A등급(이미 24시간 접근) 시설에 실내 비치      — 즉시·최저비용
#   Tier2  C등급 공공시설(주민센터·경로당 등)에 옥외함    — 24시간화, 관리주체 확보
#   Tier3  시설 자체가 없는 고위험 구역에 독립 스테이션   — 신설 필요
def greedy(cand_idx, remain, n, tier, label):
    """cand_idx 후보군에서 잔여수요를 가장 많이 덮는 지점을 순차 선택"""
    out=[]
    for _ in range(n):
        best, best_gain = None, 0.0
        for i in cand_idx:
            if i in [o["_i"] for o in out]: continue
            g = remain[cov[i]].sum()
            if g > best_gain: best, best_gain = i, g
        if best is None or best_gain <= 0: break
        got = cov[best] & (remain > 0)
        out.append(dict(_i=best, 유형=tier, 설치형태=label,
            이름=poi.이름[best], 종류=poi.종류[best], 야간등급=poi.야간등급[best],
            lon=round(poi.lon[best],6), lat=round(poi.lat[best],6),
            표고_m=round(float(poi.elev[best]),1),
            신규커버_집계구=int(got.sum()),
            신규커버_75세이상=int(dem.loc[got,"75세이상인구_2026"].sum()),
            커버_행정동="·".join(sorted(dem.loc[got,"행정동"].unique())),
            확보수요=round(float(remain[got].sum()),4)))
        remain[got] = 0.0
    return out

remain = dem["잔여수요"].values.copy()
total  = remain.sum()
rows   = []

idxA = [i for i in range(len(poi)) if poi.야간등급[i]=="A"]
idxC = [i for i in range(len(poi)) if poi.야간등급[i]=="C" and poi.종류[i] in
        ("townhall","community_centre","social_facility","library","government")]

rows += greedy(idxA, remain, N_NEW, "Tier1", "기존 24시간 시설 실내 비치")
n1 = len(rows)
rows += greedy(idxC, remain, N_NEW - n1, "Tier2", "공공시설 옥외 AED함(24시간화)")
n2 = len(rows) - n1

# Tier3 — 남은 고위험 집계구에 독립 스테이션(집계구 중심)
left = np.argsort(-remain)
for j in left[: max(0, N_NEW - len(rows))]:
    if remain[j] <= 0: break
    r = dem.iloc[j]
    rows.append(dict(_i=-1, 유형="Tier3", 설치형태="옥외 독립 AED 스테이션 신설",
        이름=f"{r.행정동} {r.집계구코드[-4:]} 구역", 종류="신설", 야간등급="A(신설)",
        lon=round(r.lon,6), lat=round(r.lat,6), 표고_m=round(float(r.elev),1),
        신규커버_집계구=1, 신규커버_75세이상=int(r["75세이상인구_2026"]),
        커버_행정동=r.행정동, 확보수요=round(float(remain[j]),4)))
    remain[j]=0.0
n3 = len(rows) - n1 - n2

res = pd.DataFrame(rows).drop(columns=["_i"])
res.insert(0, "순위", range(1, len(res)+1))
res["누적_75세이상"] = res.신규커버_75세이상.cumsum()
res["누적_수요커버율_%"] = (res.확보수요.cumsum() / total * 100).round(1)
os.makedirs("data/output", exist_ok=True)
res.to_csv(OUT, index=False, encoding="utf-8-sig")

print(f"\n[배치 전략] Tier1 {n1}곳 · Tier2 {n2}곳 · Tier3 {n3}곳 = 총 {len(res)}곳")
print(f"[성과] 수요 {(total-remain.sum())/total*100:.1f}% 확보 · "
      f"집계구 {int(res.신규커버_집계구.sum())}개 · 75세+ {int(res.신규커버_75세이상.sum()):,}명")
print(f"저장: {OUT}\n")
print(res[["순위","유형","이름","종류","설치형태","커버_행정동","신규커버_집계구","신규커버_75세이상"]].to_string(index=False))
