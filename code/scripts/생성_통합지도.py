# -*- coding: utf-8 -*-
"""통합 지도: 집계구 위험도/인구 choropleth + 도로망 + AED(주야간) + 안전센터 + 병원 + MCLP 신규"""

# ── 저장소 루트로 이동: 실행 위치와 무관하게 data/·outputs/ 상대경로 유지 ──
import os
from pathlib import Path
for _c in [Path.cwd(), *Path.cwd().parents]:
    if (_c / "README.md").exists() and (_c / ".gitignore").exists():
        os.chdir(_c)
        break

import geopandas as gpd, pandas as pd, numpy as np, folium
import branca.colormap as cm
from folium import FeatureGroup, GeoJson, GeoJsonTooltip, CircleMarker, Marker, Icon, DivIcon

OUT  = "outputs/통합지도.html"

# ---------- 1. 집계구 (위험도 + 인구) ----------
oa = gpd.read_parquet("outputs/oa_risk.parquet").to_crs(4326)
oa["p75_2026"] = oa["p75_2026"].round(0)
oa["risk_norm"] = oa["risk_norm"].round(3)
oa["dist_station_m"] = oa["dist_station_m"].round(0)

risk_cmap = cm.LinearColormap(["#2b7a78", "#7ca23f", "#d9a441", "#f0523a"],
                              vmin=0, vmax=float(oa["risk_norm"].max()), caption="집계구 위험도 (risk_norm)")
pop_cmap  = cm.LinearColormap(["#fff3b0", "#f2b705", "#d9822b", "#8a2b06"],
                              vmin=0, vmax=float(oa["p75_2026"].max()), caption="75세+ 인구 (2026-06)")

m = folium.Map(location=[35.1175, 129.042], zoom_start=15, tiles="cartodbpositron", control_scale=True)

# 배경 타일 선택지 추가
folium.TileLayer("cartodbdark_matter", name="배경: 다크").add_to(m)
folium.TileLayer("openstreetmap", name="배경: OSM").add_to(m)

fields = ["dong", "p75_2026", "dist_station_m", "risk_norm"]
aliases = ["행정동", "75세+ 인구", "안전센터 거리(m)", "위험도"]

# (A) 위험도 choropleth
fg_risk = FeatureGroup(name="① 집계구 위험도", show=False)
GeoJson(oa, style_function=lambda f: {
            "fillColor": risk_cmap(f["properties"]["risk_norm"]),
            "color": "#ffffff", "weight": 0.6, "fillOpacity": 0.72},
        highlight_function=lambda f: {"weight": 2, "color": "#111"},
        tooltip=GeoJsonTooltip(fields=fields, aliases=aliases, localize=True, sticky=False)
        ).add_to(fg_risk)
fg_risk.add_to(m)

# (B) 인구 choropleth
fg_pop = FeatureGroup(name="② 집계구 75+ 인구", show=False)
GeoJson(oa, style_function=lambda f: {
            "fillColor": pop_cmap(f["properties"]["p75_2026"]),
            "color": "#ffffff", "weight": 0.6, "fillOpacity": 0.75},
        highlight_function=lambda f: {"weight": 2, "color": "#111"},
        tooltip=GeoJsonTooltip(fields=fields, aliases=aliases, localize=True)
        ).add_to(fg_pop)
fg_pop.add_to(m)

# (라벨) 각 집계구 폴리곤에 75+ 인구수 표기
fg_lbl = FeatureGroup(name="집계구 75+ 인구수 라벨", show=True)
for (_, row), pt in zip(oa.iterrows(), oa.geometry.representative_point()):
    n = int(round(row["p75_2026"]))
    Marker([pt.y, pt.x],
           icon=DivIcon(icon_size=(0, 0), icon_anchor=(0, 0),
                        html=f'<div style="font-size:11px;font-weight:800;color:#141414;'
                             f'text-shadow:0 0 2px #fff,0 0 2px #fff,0 0 3px #fff,0 0 3px #fff;'
                             f'transform:translate(-50%,-50%);white-space:nowrap">{n}</div>')
           ).add_to(fg_lbl)
fg_lbl.add_to(m)

# ---------- 2. 도로망 (방향별 부호경사) ----------
from shapely import offset_curve
ep = gpd.read_file("outputs/graph_drive_conn.gpkg", layer="edges")
if ep.crs is None:
    ep.set_crs(4326, inplace=True)
ep = ep.to_crs(5186)                                   # 미터 좌표계에서 오프셋

# ★ 표시 범위 축소: 분석 그래프는 그대로 두고, 지도에는 집계구 영역 + 400m 버퍼만 클립
study_buf = oa.to_crs(5186).geometry.unary_union.buffer(400)
ep = gpd.clip(ep, study_buf)

ep["grade"] = pd.to_numeric(ep["grade"], errors="coerce").fillna(0)   # 방향별 부호경사(진행방향 오르막 +, 내리막 −)
ep["grade_abs"] = ep["grade"].abs()
ep["grade_pct"] = (ep["grade"] * 100).round(1)
# 절대 표고(고도): 국소 경사가 평탄해도 '높은 곳'임을 드러내기 위함
ep["elev_mid"] = ((pd.to_numeric(ep["elev_u"], errors="coerce")
                   + pd.to_numeric(ep["elev_v"], errors="coerce")) / 2).fillna(0).round(0)

# === 연속 표시용: 도로 1개당 한 선(무방향, 중심선) — 노드에서 실제로 이어져 틈 없음 ===
ep["pair"] = [tuple(sorted((u, v))) for u, v in zip(ep["u"], ep["v"])]
cont = ep.sort_values("grade_abs", ascending=False).drop_duplicates("pair").to_crs(4326)  # 쌍당 급한 쪽 유지
cont["grade_pct_abs"] = (cont["grade_abs"] * 100).round(1)

# === 방향별용: 진행방향 옆으로 1.2m 오프셋 (오르막/내리막 색 구분, 겹침 방지) ===
def off_right(g):
    try:
        o = offset_curve(g, -1.2)
        return g if (o is None or o.is_empty) else o
    except Exception:
        return g
epd = ep.copy()
epd["geometry"] = epd.geometry.apply(off_right)
edges_dir = epd.to_crs(4326)

# (③) 단색 도로망 (연속·기본 꺼짐)
fg_road = FeatureGroup(name="③ 차량 도로망 (단색)", show=True)
GeoJson(cont[["geometry"]], style_function=lambda f: {
            "color": "#4a6b8a", "weight": 1.2, "opacity": 0.6,
            "lineCap": "round", "lineJoin": "round"}).add_to(fg_road)
fg_road.add_to(m)

# (③b) 도로 경사 — 연속(중심선, 틈 없음), 절대경사 |grade| 로 급경사 강조 (기본 켜짐)
grade_abs_cmap = cm.LinearColormap(
    ["#2c7fb8", "#7fcdbb", "#c7e9b4", "#fed976", "#fd8d3c", "#e31a1c"],
    index=[0, 0.05, 0.08, 0.12, 0.15, 0.25], vmin=0, vmax=0.25,
    caption="도로 경사 |grade| (완만 → 급경사, 15%↑ 빨강) · 연속 표시")
fg_grade = FeatureGroup(name="③b 도로 경사 (연속·급경사 강조)", show=False)
GeoJson(cont[["grade_pct_abs", "grade_abs", "geometry"]],
        style_function=lambda f: {
            "color": grade_abs_cmap(min(f["properties"]["grade_abs"], 0.25)),
            "weight": 3, "opacity": 0.9, "lineCap": "round", "lineJoin": "round"},
        tooltip=GeoJsonTooltip(fields=["grade_pct_abs"], aliases=["도로 경사(%)"], localize=True)
        ).add_to(fg_grade)
fg_grade.add_to(m)

# (③c) 도로 경사 — 방향별 부호(오프셋), 내리막 파랑 ↔ 오르막 빨강 (기본 꺼짐)
grade_cmap = cm.LinearColormap(
    ["#2166ac", "#67a9cf", "#d1e5f0", "#f7f7f7", "#fddbc7", "#ef8a62", "#b2182b"],
    index=[-0.25, -0.15, -0.05, 0, 0.05, 0.15, 0.25], vmin=-0.25, vmax=0.25,
    caption="도로 경사(방향별): 내리막(−) 파랑 ↔ 오르막(+) 빨강 · 오프셋 표시")
fg_gdir = FeatureGroup(name="③c 도로 경사 (방향별 부호·오프셋)", show=False)
GeoJson(edges_dir[["grade_pct", "grade", "geometry"]],
        style_function=lambda f: {
            "color": grade_cmap(max(-0.25, min(0.25, f["properties"]["grade"]))),
            "weight": 2.4, "opacity": 0.9},
        tooltip=GeoJsonTooltip(fields=["grade_pct"], aliases=["진행방향 경사(%, +오르막/−내리막)"], localize=True)
        ).add_to(fg_gdir)
fg_gdir.add_to(m)

# (③d) 도로 표고(고도) — 국소 경사가 평탄해도 '높은 곳'을 빨강으로 드러냄 (기본 켜짐)
elev_cmap = cm.LinearColormap(
    ["#3b7dd8", "#63c6a8", "#c7e9b4", "#fed976", "#d98a2b", "#8a2b06"],
    index=[0, 30, 60, 100, 150, 251], vmin=0, vmax=251,
    caption="도로 표고(고도) m — 높을수록 도달 불리 (평탄부여도 고지대면 빨강)")
fg_elev = FeatureGroup(name="③d 도로 표고 (고도)", show=False)
GeoJson(cont[["elev_mid", "geometry"]],
        style_function=lambda f: {
            "color": elev_cmap(min(max(f["properties"]["elev_mid"], 0), 251)),
            "weight": 3, "opacity": 0.9, "lineCap": "round", "lineJoin": "round"},
        tooltip=GeoJsonTooltip(fields=["elev_mid"], aliases=["표고(m)"], localize=True)
        ).add_to(fg_elev)
fg_elev.add_to(m)

# (③e) 표고 격자 (DEM) — 노드 표고를 75m 격자에 보간 → 지역 전체를 '면'으로 표시 (기본 켜짐)
import numpy as np
from shapely.geometry import box
from scipy.interpolate import griddata
nodes = gpd.read_parquet("outputs/nodes_drive_conn.parquet")   # 4,756 노드(EPSG:5186, elev)
npts = np.c_[nodes.geometry.x.values, nodes.geometry.y.values]
nval = pd.to_numeric(nodes["elev"], errors="coerce").values
minx, miny, maxx, maxy = study_buf.bounds
CELL = 75                                                              # 75m 격자
gx = np.arange(minx, maxx, CELL); gy = np.arange(miny, maxy, CELL)
polys, ctr = [], []
for x0 in gx:
    for y0 in gy:
        polys.append(box(x0, y0, x0 + CELL, y0 + CELL)); ctr.append((x0 + CELL / 2, y0 + CELL / 2))
ctr = np.array(ctr)
grid = gpd.GeoDataFrame({"geometry": polys}, crs=5186)
ev = griddata(npts, nval, ctr, method="linear")                       # 선형 보간
nanm = np.isnan(ev)
ev[nanm] = griddata(npts, nval, ctr[nanm], method="nearest")          # 볼록껍질 밖은 최근접으로 채움
grid["elev"] = np.round(ev, 0)
grid = grid[grid.intersects(study_buf)].copy()                        # 스터디 영역만
grid_ll = grid.to_crs(4326)
fg_dem = FeatureGroup(name="③e 표고 격자 (DEM 75m)", show=True)
GeoJson(grid_ll[["elev", "geometry"]],
        style_function=lambda f: {
            "fillColor": elev_cmap(min(max(f["properties"]["elev"], 0), 251)),
            "color": "#00000000", "weight": 0, "fillOpacity": 0.55},
        tooltip=GeoJsonTooltip(fields=["elev"], aliases=["표고(m)"], localize=True)
        ).add_to(fg_dem)
fg_dem.add_to(m)

# ---------- 3. AED (주간/야간) ----------
aed = pd.read_csv("outputs/aed_donggu.csv")
def is_night(row):
    s = str(row.get("monSttTme", "")).strip().replace(".0", "").zfill(4)
    e = str(row.get("monEndTme", "")).strip().replace(".0", "").zfill(4)
    try: en = int(e)
    except: en = -1
    return (en >= 2200) or (s == "0000" and e in ("0000", "2400"))
aed["night"] = aed.apply(is_night, axis=1)
aed = aed.dropna(subset=["wgs84Lat", "wgs84Lon"])

fg_aed_n = FeatureGroup(name="④ AED · 야간 접근 (60)", show=True)
fg_aed_d = FeatureGroup(name="⑤ AED · 주간 전용 (46)", show=False)
for _, r in aed.iterrows():
    # 개인정보(manager/tel) 제외, 위치·시간만 표시
    popup = folium.Popup(
        f"<b>{r.get('org','')}</b><br>{r.get('buildPlace','')}<br>"
        f"월 {str(r.get('monSttTme','')).zfill(4)}~{str(r.get('monEndTme','')).zfill(4)} · "
        f"{'야간O' if r['night'] else '주간전용'}", max_width=240)
    tgt = fg_aed_n if r["night"] else fg_aed_d
    CircleMarker([r["wgs84Lat"], r["wgs84Lon"]], radius=4,
                 color="#1f78b4" if r["night"] else "#b0b0b0",
                 fill=True, fill_color="#2fa5c7" if r["night"] else "#cccccc",
                 fill_opacity=0.9, weight=1, popup=popup).add_to(tgt)
fg_aed_n.add_to(m); fg_aed_d.add_to(m)

# ---------- 4. 안전센터 7곳 (좌표·명칭 하드코딩: 검증 완료본) ----------
centers = [
    ("초량119안전센터", "중부소방서", 35.11974, 129.04294, "0m"),
    ("부두119안전센터", "항만소방서", 35.12544, 129.05070, "80m"),
    ("수정119안전센터", "부산진소방서", 35.12922, 129.04917, "125m"),
    ("중앙119안전센터(중부)", "중부소방서", 35.10719, 129.03615, "484m"),
    ("범일119안전센터", "부산진소방서", 35.14020, 129.05938, "493m"),
    ("창선119안전센터", "중부소방서", 35.10266, 129.02726, "1,413m"),
    ("청학119안전센터", "항만소방서", 35.09831, 129.05640, "1,443m"),
]
fg_ctr = FeatureGroup(name="⑥ 119안전센터 7곳 (구급차 7/7)", show=True)
for nm, so, lat, lon, dm in centers:
    Marker([lat, lon], icon=Icon(color="red", icon="plus-sign"),
           popup=folium.Popup(f"<b>{nm}</b><br>{so} · 구급차 1대<br>대상지 {dm}", max_width=220)).add_to(fg_ctr)
fg_ctr.add_to(m)

# ---------- 5. 이송병원 3곳 ----------
hosp = [
    ("동아대학교병원", "권역응급의료센터", 35.120006, 129.017604),
    ("부산대학교병원", "지역응급의료센터", 35.101054, 129.019222),
    ("인제대부산백병원", "지역응급의료센터", 35.146454, 129.020572),
]
fg_h = FeatureGroup(name="⑦ 이송병원 3곳", show=True)
for nm, gr, lat, lon in hosp:
    Marker([lat, lon], icon=Icon(color="darkblue", icon="h-square", prefix="fa"),
           popup=folium.Popup(f"<b>{nm}</b><br>{gr}", max_width=220)).add_to(fg_h)
fg_h.add_to(m)

# ---------- 6. MCLP 신규 후보 15개 ----------
mclp = pd.read_csv("outputs/oa_mclp_new_aed.csv")
fg_m = FeatureGroup(name="⑧ MCLP 신규 AED 후보 (15)", show=True)
for _, r in mclp.iterrows():
    Marker([r["lat"], r["lon"]],
           icon=DivIcon(html=f'<div style="font-size:20px;color:#16a34a;'
                             f'text-shadow:0 0 3px #fff,0 0 3px #fff">★</div>'),
           popup=folium.Popup(f"신규#{int(r['rank'])} · {r['dong']}<br>75+ {int(r['p75_2026'])} · 위험 {r['risk_norm']:.2f}",
                              max_width=200)).add_to(fg_m)
fg_m.add_to(m)

# ---------- 6b. AED 후보지 선정 알고리즘 시각화 (MCLP 커버리지) ----------
# 야간 공백 집계구(고위험 & 야간 미커버) = MCLP의 '수요' → 신규 후보가 이걸 덮음
try:
    gap = gpd.read_parquet("outputs/oa_gap.parquet").to_crs(4326)
    gapf = gap[gap["gap"] == True] if "gap" in gap.columns else gap.iloc[0:0]
    fg_gap = FeatureGroup(name="⑨ 야간 공백 집계구 (MCLP 수요)", show=True)
    GeoJson(gapf[["geometry"]], style_function=lambda f: {
                "color": "#b2182b", "weight": 2.4, "fill": True,
                "fillColor": "#b2182b", "fillOpacity": 0.20}).add_to(fg_gap)
    fg_gap.add_to(m)
    print(f"공백 집계구(수요): {len(gapf)}")
except Exception as e:
    print("gap 레이어 경고:", e)

# 신규 후보 커버 반경 150m — MCLP가 새로 덮는 범위(초록 원)
fg_cov = FeatureGroup(name="⑩ 신규 후보 커버 (R=150m)", show=True)
for _, r in mclp.iterrows():
    folium.Circle([r["lat"], r["lon"]], radius=150, color="#16a34a", weight=1.5,
                  fill=True, fill_color="#16a34a", fill_opacity=0.12,
                  tooltip=f"신규#{int(r['rank'])} · {r['dong']} 커버 150m").add_to(fg_cov)
fg_cov.add_to(m)

# 기존 야간 AED 커버 반경 150m — 이미 커버된 곳(공백과 대비, 저지대 편중 확인)
fg_ncov = FeatureGroup(name="⑪ 기존 야간AED 커버 (R=150m)", show=False)
for _, r in aed[aed["night"]].iterrows():
    folium.Circle([r["wgs84Lat"], r["wgs84Lon"]], radius=150, color="#1f78b4", weight=1,
                  fill=True, fill_color="#1f78b4", fill_opacity=0.08).add_to(fg_ncov)
fg_ncov.add_to(m)

# ---------- 마무리: 범례 + 레이어 컨트롤 + 제목 ----------
risk_cmap.add_to(m); grade_abs_cmap.add_to(m); elev_cmap.add_to(m)
folium.LayerControl(collapsed=False).add_to(m)

title_html = '''
<div style="position:fixed;top:12px;left:50%;transform:translateX(-50%);z-index:9999;
     background:rgba(20,31,30,.92);color:#e9edea;padding:9px 16px;border-radius:10px;
     font-family:'Malgun Gothic',sans-serif;box-shadow:0 4px 18px rgba(0,0,0,.35);text-align:center">
  <div style="font-size:14px;font-weight:800">부산 동구 산복도로 AED 최적배치 — 통합 지도</div>
  <div style="font-size:11px;color:#93a6a1;margin-top:2px">
    집계구 78 · 위험도/인구 · 도로망 · AED 106(야60/주46) · 안전센터 7 · 병원 3 · <b>MCLP 신규 15 + 커버리지</b>
    &nbsp;|&nbsp; 우측 상단 ☰ 에서 레이어 On/Off</div>
</div>'''
m.get_root().html.add_child(folium.Element(title_html))

# 좌측 상단: 레이어 설명 패널 (접기/펼치기)
legend_html = '''
<div style="position:fixed;top:78px;left:10px;z-index:9999;max-width:270px;
     font-family:'Malgun Gothic',sans-serif;">
 <details open style="background:rgba(255,255,255,.95);border:1px solid #cbd5d1;border-radius:10px;
     box-shadow:0 3px 14px rgba(0,0,0,.18);padding:8px 12px 10px;color:#1a2422;">
  <summary style="cursor:pointer;font-size:12.5px;font-weight:800;outline:none;">📖 레이어 설명 (클릭해 접기)</summary>
  <div style="font-size:11px;line-height:1.55;margin-top:7px;max-height:56vh;overflow:auto;">
   <div style="font-weight:700;color:#b2182b;margin:2px 0">▮ 수요 · 위험</div>
   ① 집계구 위험도 = 75+인구 × 도달지연 (빨강=고위험)<br>
   ② 집계구 75+ 인구 (노랑→갈색)<br>
   · 인구수 라벨 = 각 집계구 75세+ 명수<br>
   <div style="font-weight:700;color:#8a2b06;margin:6px 0 2px">▮ 도로 · 지형</div>
   ③ 도로망(단색) — 위치 확인용<br>
   ③b 도로 경사(연속) = 국소 기울기, 15%↑ 빨강<br>
   ③c 도로 경사(방향별) = 오르막 빨강/내리막 파랑<br>
   <b>③d 도로 표고</b> = 도로의 고도<br>
   <b>③e 표고 격자(DEM 75m)</b> = 지역 전체 고도 면<br>
   <span style="color:#555">↳ ③d·③e는 <b>표고(고도)</b> 레이어 — 평탄부여도 고지대면 빨강.
     체크박스로 <b>껐다 켰다</b> 하세요.</span><br>
   <div style="font-weight:700;color:#1f78b4;margin:6px 0 2px">▮ 시설</div>
   ④ AED 야간접근(파랑) / ⑤ AED 주간전용(회색)<br>
   ⑥ 119안전센터 7곳(빨강 십자) · ⑦ 병원 3곳(파란 H)<br>
   ⑧ MCLP 신규 AED 후보 15(초록 별)<br>
   <div style="font-weight:700;color:#16a34a;margin:6px 0 2px">▮ AED 후보지 알고리즘 (MCLP)</div>
   ⑨ 야간 공백 집계구(빨강 면) = <b>수요</b>(고위험 & 야간 미커버)<br>
   ⑩ 신규 후보 커버 150m(초록 원) = MCLP가 새로 덮는 범위<br>
   ⑪ 기존 야간AED 커버 150m(파랑 원) = 이미 커버된 곳<br>
   <span style="color:#555">↳ 빨강 공백을 초록 원이 덮도록 후보를 고르는 게 MCLP.</span><br>
   <div style="margin-top:7px;color:#555;border-top:1px solid #e0e6e4;padding-top:6px">
    · 우측 상단 <b>☰ 체크박스</b>로 레이어 On/Off<br>
    · 우측 하단 <b>색막대</b> = 위험도·경사·표고 범례<br>
    · 도로/지형 색 레이어는 <b>한 번에 하나</b>만 켜야 선명합니다</div>
  </div>
 </details>
</div>'''
m.get_root().html.add_child(folium.Element(legend_html))

m.save(OUT)
print("saved:", OUT)
print("AED night/day:", int(aed["night"].sum()), "/", int((~aed["night"]).sum()))
print("도로(연속):", len(cont), "| 도로(방향별):", len(edges_dir), "| oa:", len(oa))
