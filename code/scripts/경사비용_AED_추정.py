# -*- coding: utf-8 -*-
"""경사비용(오르막 페널티) 구현·검증 + 그 위에서 AED 후보 재산출 (실측 수치 확보용)"""
import sys; sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd, geopandas as gpd, networkx as nx, osmnx as ox
from scipy.spatial import cKDTree

BASE = r"C:\Users\doyun\켠김에왕까지_부산대이데대회"
BETA = 6.0; R = 150; N_NEW = 15

def _sf(x):
    try: return float(x)
    except: return float("nan")

# ---------- 1. 그래프 + 경사비용 엣지 가중 ----------
G = ox.load_graphml(BASE + r"\outputs\graph_drive_conn.graphml",
                    edge_dtypes={"length": float, "grade": _sf}, node_dtypes={"elev": _sf})
n_up = 0
for u, v, k, d in G.edges(keys=True, data=True):
    L = float(d.get("length", 0) or 0)
    g = d.get("grade"); g = g if (isinstance(g, float) and g == g) else 0.0
    d["cost"] = L * (1.0 + BETA * max(0.0, g))     # 오르막(진행방향 +경사)만 페널티
    if g > 0: n_up += 1
print(f"[경사비용] β={BETA} · 오르막 엣지 {n_up}/{G.number_of_edges()} 에 페널티 적용")

stations = [n for n, d in G.nodes(data=True) if d.get("node_type") == "station"]
print(f"안전센터 노드: {len(stations)}")

def multi_source_min(weight):
    best = {}
    for s in stations:
        for n, dd in nx.single_source_dijkstra_path_length(G, s, weight=weight).items():
            if n not in best or dd < best[n]: best[n] = dd
    return best
best_len   = multi_source_min("length")   # 기존(거리)
best_cost  = multi_source_min("cost")      # 신규(경사비용 유효거리)

# ---------- 2. 집계구에 부여 + 위험도 재계산 ----------
oa = gpd.read_parquet(BASE + r"\outputs\oa_risk.parquet")
oa["reach_len"]   = oa["entry_node"].map(lambda n: best_len.get(n, np.nan))
oa["reach_slope"] = oa["entry_node"].map(lambda n: best_cost.get(n, np.nan))
oa["slope_ratio"] = oa["reach_slope"] / oa["reach_len"]     # 경사로 인한 유효거리 증가배율

def nrm(s):
    s = pd.to_numeric(s, errors="coerce")
    return (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else s * 0

cap = oa["reach_slope"].clip(upper=oa["reach_slope"].quantile(0.95))
oa["f_reach_slope"] = nrm(cap)
oa["risk_slope"] = nrm(oa["f_age"] * oa["f_reach_slope"])   # f_age(75+ 정규화)는 그대로

print("\n[유효거리 증가배율 slope_ratio] 평균 %.2f · 중앙 %.2f · 최대 %.2f"
      % (oa["slope_ratio"].mean(), oa["slope_ratio"].median(), oa["slope_ratio"].max()))
print("\n[동별 위험도: 거리기준 → 경사비용]")
cmp = oa.groupby("dong").agg(거리=("risk_norm", "mean"), 경사=("risk_slope", "mean")).round(3)
cmp["Δ"] = (cmp["경사"] - cmp["거리"]).round(3)
print(cmp.sort_values("경사", ascending=False).to_string())

print("\n[집계구 위험순위 상위8: 거리 vs 경사]")
a = oa.sort_values("risk_norm", ascending=False).head(8)[["dong", "TOT_OA_CD"]].reset_index(drop=True)
b = oa.sort_values("risk_slope", ascending=False).head(8)[["dong", "TOT_OA_CD"]].reset_index(drop=True)
for i in range(8):
    print(f"  {i+1}. 거리:{a.loc[i,'dong']}({a.loc[i,'TOT_OA_CD']})  |  경사:{b.loc[i,'dong']}({b.loc[i,'TOT_OA_CD']})")

# ---------- 3. 경사비용 위험도 위에서 AED 후보(MCLP) 재산출 ----------
aed = pd.read_csv(BASE + r"\outputs\aed_donggu.csv").dropna(subset=["wgs84Lat", "wgs84Lon"]).copy()
ag = gpd.GeoDataFrame(aed, geometry=gpd.points_from_xy(aed["wgs84Lon"], aed["wgs84Lat"]), crs=4326).to_crs(5186)
def to_hhmm(v):
    try: return int(float(v))
    except: return None
def is_night(r):
    e = to_hhmm(r.get("monEndTme")); s = to_hhmm(r.get("monSttTme"))
    if e is None: return False
    if s == 0 and e in (0, 2400, 2359): return True
    return e >= 2200
ag["night"] = ag.apply(is_night, axis=1)

cen = np.c_[oa["cx"].values, oa["cy"].values]
npt = np.c_[ag.loc[ag["night"], "geometry"].x.values, ag.loc[ag["night"], "geometry"].y.values]
oa["aed_dist_night"] = cKDTree(npt).query(cen, k=1)[0]
oa["covered_night"] = oa["aed_dist_night"] <= R

def mclp(riskcol):
    demand = (oa[riskcol] > 0) & (~oa["covered_night"])
    d_idx = np.where(demand.values)[0]; d_w = oa[riskcol].values[d_idx]
    tree = cKDTree(cen[d_idx]); cover = tree.query_ball_point(cen, R)
    covered = np.zeros(len(d_idx), bool); selected = []; hist = []
    for _ in range(N_NEW):
        best_g, best_c = -1, -1
        for ci, dl in enumerate(cover):
            if not dl: continue
            g = d_w[[j for j in dl if not covered[j]]].sum()
            if g > best_g: best_g, best_c = g, ci
        if best_c < 0 or best_g <= 0: break
        for j in cover[best_c]: covered[j] = True
        selected.append(best_c); hist.append(d_w[covered].sum())
    tot = d_w.sum()
    return selected, [round(h / tot * 100) for h in hist], len(d_idx), tot

sel_len, cov_len, nd_len, _ = mclp("risk_norm")
sel_slp, cov_slp, nd_slp, _ = mclp("risk_slope")
print(f"\n[MCLP 커버율 곡선] 거리기준(수요{nd_len}): {cov_len}")
print(f"[MCLP 커버율 곡선] 경사기준(수요{nd_slp}): {cov_slp}")
print("\n[경사비용 기준 신규 후보 배분]")
print(oa.iloc[sel_slp]["dong"].value_counts().to_string())
print("\n[거리기준 신규 후보 배분]")
print(oa.iloc[sel_len]["dong"].value_counts().to_string())
print("\nDONE")
