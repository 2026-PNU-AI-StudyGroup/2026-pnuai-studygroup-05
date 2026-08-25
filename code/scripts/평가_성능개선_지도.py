# -*- coding: utf-8 -*-
"""07. 성능평가 커버 지도 (folium 인터랙티브) — 팀 지도 형식과 통일

배치 전(기존106)/후(통합) 야간 AED 커버를 집계구 색으로, 기존/신규 AED를 마커로.
레이어 토글로 전/후·AED 종류를 켜고 끈다.

입력: outputs/oa_risk.parquet(집계구 경계) + data/output/평가_집계구별.csv(커버결과)
      outputs/aed_donggu.csv(기존, 있으면) · data/output/AED_신규배치_접근성.csv(신규)
출력: results/평가_커버지도.html
경로: 저장소 루트 기준 상대경로.
"""
import sys, os
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
for _c in [Path.cwd(), *Path.cwd().parents]:
    if (_c / "README.md").exists() and (_c / ".gitignore").exists():
        os.chdir(_c); break

import numpy as np, pandas as pd, geopandas as gpd, folium
from folium import FeatureGroup, GeoJson, GeoJsonTooltip, CircleMarker, Marker, Icon, DivIcon

if not Path("outputs/oa_risk.parquet").exists():
    raise SystemExit("outputs/oa_risk.parquet 없음(집계구 경계) → 노트북 10 먼저 실행. (저장소엔 산출물 미포함)")

# ── 집계구 + 커버결과 ──
oa = gpd.read_parquet("outputs/oa_risk.parquet")[["TOT_OA_CD","dong","p75_2026","geometry"]].to_crs(4326)
oa["TOT_OA_CD"] = oa["TOT_OA_CD"].astype(str)
ev = pd.read_csv("data/output/평가_집계구별.csv", encoding="utf-8-sig")
ev["TOT_OA_CD"] = ev["TOT_OA_CD"].astype(str)
g = oa.merge(ev, on="TOT_OA_CD", suffixes=("", "_e"))

TEAL, ACC, BLUE, GREEN = "#2c7a7b", "#c0392b", "#1f78b4", "#16a34a"
m = folium.Map(location=[35.117, 129.042], zoom_start=15, tiles="cartodbpositron", control_scale=True)
folium.TileLayer("cartodbdark_matter", name="배경: 다크").add_to(m)

def cover_layer(col, name, show):
    fg = FeatureGroup(name=name, show=show)
    GeoJson(g[["geometry", col, "dong", "p75_2026", "구급차도달_분"]],
            style_function=lambda f, c=col: {
                "fillColor": TEAL if f["properties"][c] else ACC,
                "color": "white", "weight": 0.5,
                "fillOpacity": 0.7 if f["properties"][c] else 0.45},
            tooltip=GeoJsonTooltip(
                fields=["dong", "p75_2026", "구급차도달_분", col],
                aliases=["동", "75세+(명)", "구급차도달(분)", "야간커버"], localize=True)
            ).add_to(fg)
    return fg

# ① 배치 전 (기존 106) ── ② 배치 후 (통합)
n_bef = int(g["커버_기존야간"].sum()); p_bef = g.loc[g["커버_기존야간"], "p75_2026"].sum()
n_aft = int(g["커버_통합야간"].sum()); p_aft = g.loc[g["커버_통합야간"], "p75_2026"].sum()
cover_layer("커버_기존야간", f"① 배치 전(기존 106) — {n_bef}/78, 75+ {p_bef:,.0f}명", False).add_to(m)
cover_layer("커버_통합야간", f"② 배치 후(통합) — {n_aft}/78, 75+ {p_aft:,.0f}명", True).add_to(m)

# ③ 신규가 새로 커버한 집계구 강조
fg_new = FeatureGroup(name="③ 신규가 새로 커버(강조)", show=True)
GeoJson(g.loc[g["신규커버_획득"], ["geometry", "dong"]],
        style_function=lambda f: {"fillColor": "#f1c40f", "color": "#b8860b",
                                  "weight": 1.5, "fillOpacity": 0.5}).add_to(fg_new)
fg_new.add_to(m)

# ④ 기존 AED (야간, 있으면) ── ⑤ 신규 후보(Tier별)
if Path("outputs/aed_donggu.csv").exists():
    aed = pd.read_csv("outputs/aed_donggu.csv", encoding="utf-8-sig").dropna(subset=["wgs84Lat","wgs84Lon"])
    def night(r):
        try: e = int(float(r.get("monEndTme")))
        except: return False
        s = str(r.get("monSttTme","")).strip().replace(".0","")
        return e >= 2200 or (s in ("0","0000","") and e in (0,2400,2359))
    aed["night"] = aed.apply(night, axis=1)
    fg_ae = FeatureGroup(name="④ 기존 AED · 야간접근(60)", show=False)
    for _, r in aed[aed.night].iterrows():
        CircleMarker([r.wgs84Lat, r.wgs84Lon], radius=3, color=BLUE, fill=True,
                     fill_opacity=0.8, weight=1,
                     popup=folium.Popup(f"{r.get('org','')}<br>{r.get('buildPlace','')}", max_width=200)).add_to(fg_ae)
    fg_ae.add_to(m)

nw = pd.read_csv("data/output/AED_신규배치_접근성.csv", encoding="utf-8-sig")
GOLD = "#b8860b"
tier_col = {"Tier1": GREEN, "Tier2": GOLD, "Tier3": ACC}
fg_nw = FeatureGroup(name="⑤ 신규 후보 36(Tier별)", show=True)
for _, r in nw.iterrows():
    c = tier_col.get(r["유형"], GREEN)
    Marker([r.lat, r.lon],
           icon=DivIcon(html=f'<div style="font-size:16px;color:{c};text-shadow:0 0 3px #fff,0 0 3px #fff">★</div>'),
           popup=folium.Popup(f"[{r['유형']}] {r['이름']}<br>{r.get('설치형태','')}", max_width=220)).add_to(fg_nw)
fg_nw.add_to(m)

folium.LayerControl(collapsed=False).add_to(m)

title = '''<div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);z-index:9999;
 background:rgba(20,31,30,.92);color:#e9edea;padding:8px 15px;border-radius:9px;
 font-family:'Malgun Gothic',sans-serif;box-shadow:0 3px 14px rgba(0,0,0,.35);text-align:center">
 <b style="font-size:13px">야간 AED 커버 — 배치 전/후 (성능평가)</b><br>
 <span style="font-size:10.5px;color:#93a6a1">청록=커버 · 빨강=공백 · 노랑=신규가 새로 커버 · ★=신규후보(초록T1/금T2/빨강T3)</span></div>'''
m.get_root().html.add_child(folium.Element(title))

legend = '''<div style="position:fixed;bottom:22px;left:12px;z-index:9999;background:rgba(255,255,255,.95);
 padding:8px 11px;border:1px solid #cbd5d1;border-radius:8px;font-size:11px;font-family:'Malgun Gothic',sans-serif">
 <b>성능평가 요약(야간)</b><br>
 배치 전 24/78 · 75+ 1,294명<br>
 배치 후 61/78 · 75+ 4,028명 <b style="color:#c0392b">(+2,734)</b><br>
 8분초과 취약지 9%→91%</div>'''
m.get_root().html.add_child(folium.Element(legend))

Path("results").mkdir(exist_ok=True)
m.save("results/평가_커버지도.html")
print("저장: results/평가_커버지도.html")
