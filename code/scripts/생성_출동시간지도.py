# -*- coding: utf-8 -*-
"""출동시간 지도: 집계구 도달시간·지연 choropleth + 구간별 분해 + 안전센터·병원

`출동시간_추정.py`의 결과(outputs/oa_delay.csv)를 집계구 경계에 붙여 지도로 낸다.
docs/04 의 통합지도와 같은 folium 레이어 방식이며, 도달'거리'가 아니라
도달'시간'과 '지연'을 보여주는 것이 차이다.

실행:  python code/scripts/생성_출동시간지도.py
출력:  demo/출동시간지도.html
"""
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

# ── 저장소 루트로 이동 ──
for _c in [Path.cwd(), *Path.cwd().parents]:
    if (_c / "README.md").exists() and (_c / ".gitignore").exists():
        os.chdir(_c)
        break

import geopandas as gpd
import pandas as pd
import folium
import branca.colormap as cm
from folium import FeatureGroup, GeoJson, GeoJsonTooltip, Marker, Icon

OUT = "demo/출동시간지도.html"
HOSPITALS = [("동아대학교병원", "권역", 129.017604, 35.120006),
             ("부산대학교병원", "지역", 129.019222, 35.101054),
             ("인제대학교부산백병원", "지역", 129.020572, 35.146454)]


def main():
    oa = gpd.read_parquet("outputs/oa_risk.parquet").to_crs(4326)
    d = pd.read_csv("outputs/oa_delay.csv", encoding="utf-8-sig")
    d["TOT_OA_CD"] = d.TOT_OA_CD.astype(str)
    oa["TOT_OA_CD"] = oa.TOT_OA_CD.astype(str)
    keep = ["TOT_OA_CD", "t1_min", "t2_min", "t3_min", "total_min",
            "delay_min", "delay_t2_min", "risk_norm_time", "access_dx_m"]
    g = oa[["TOT_OA_CD", "dong", "p75_2026", "geometry"]].merge(d[keep], on="TOT_OA_CD")
    for c in ["t1_min", "t2_min", "t3_min", "total_min", "delay_min",
              "delay_t2_min", "access_dx_m"]:
        g[c] = g[c].round(2)
    g["p75_2026"] = g.p75_2026.round(0)
    g["risk_norm_time"] = g.risk_norm_time.round(3)

    m = folium.Map(location=[35.1175, 129.042], zoom_start=15,
                   tiles="cartodbpositron", control_scale=True)
    folium.TileLayer("cartodbdark_matter", name="배경: 다크").add_to(m)
    folium.TileLayer("openstreetmap", name="배경: OSM").add_to(m)

    fields = ["dong", "p75_2026", "t1_min", "t2_min", "t3_min",
              "total_min", "delay_min", "risk_norm_time"]
    aliases = ["행정동", "75세+ 인구", "t1 출동(분)", "t2 들것(분)", "t3 이송(분)",
               "총 도달시간(분)", "지연(분)", "위험도(시간기준)"]

    # 레이어 정의: (표시명, 컬럼, 색상, 설명, 기본표시)
    layers = [
        ("① 총 도달시간(분)", "total_min",
         ["#2c7bb6", "#abd9e9", "#fdae61", "#d7191c"], "총 도달시간 (분)", True),
        ("② 지연 = 평지 대비 손해(분)", "delay_min",
         ["#f7f7f7", "#fdd49e", "#fc8d59", "#b30000"], "지연 (분)", False),
        ("③ t1 출동시간(분)", "t1_min",
         ["#f7fcf5", "#bae4b3", "#74c476", "#238b45"], "t1 출동 (분)", False),
        ("④ t2 들것 도보(분)", "t2_min",
         ["#fff7fb", "#d0d1e6", "#74a9cf", "#0570b0"], "t2 들것 도보 (분)", False),
        ("⑤ t3 이송(분)", "t3_min",
         ["#fcfbfd", "#dadaeb", "#9e9ac8", "#6a51a3"], "t3 이송 (분)", False),
        ("⑥ 위험도(시간기준)", "risk_norm_time",
         ["#2b7a78", "#7ca23f", "#d9a441", "#f0523a"], "위험도 (시간기준)", False),
    ]

    for name, col, colors, caption, show in layers:
        cmap = cm.LinearColormap(colors, vmin=float(g[col].min()),
                                 vmax=float(g[col].max()), caption=caption)
        fg = FeatureGroup(name=name, show=show)
        GeoJson(g, style_function=lambda f, c=col, k=cmap: {
                    "fillColor": k(f["properties"][c]),
                    "color": "#ffffff", "weight": 0.6, "fillOpacity": 0.75},
                highlight_function=lambda f: {"weight": 2, "color": "#111"},
                tooltip=GeoJsonTooltip(fields=fields, aliases=aliases, localize=True)
                ).add_to(fg)
        fg.add_to(m)
        if show:
            cmap.add_to(m)

    # 안전센터 7곳 (출동 출발지)
    fg_st = FeatureGroup(name="⑦ 119안전센터 7곳", show=True)
    sc = pd.read_csv("outputs/safety_centers.csv", encoding="utf-8-sig")
    sc.columns = ["name", "lon", "lat", "dist_m", "region",
                  "osm_name", "fire_dept", "amb_count", "verify"]
    for r in sc.itertuples():
        Marker([r.lat, r.lon], tooltip=f"{r.name} · 구급차 {r.amb_count}대 · {r.fire_dept}",
               icon=Icon(color="blue", icon="truck-medical", prefix="fa")).add_to(fg_st)
    fg_st.add_to(m)

    # 이송 병원 3곳
    fg_hp = FeatureGroup(name="⑧ 이송 병원 3곳", show=True)
    for nm, gr, lo, la in HOSPITALS:
        Marker([la, lo], tooltip=f"{nm} ({gr}응급의료센터)",
               icon=Icon(color="red", icon="plus", prefix="fa")).add_to(fg_hp)
    fg_hp.add_to(m)

    legend = (
        '<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:#fff;'
        'padding:10px 14px;border:1px solid #999;border-radius:4px;font-size:12px;'
        'max-width:290px;line-height:1.55">'
        '<b>출동시간 · 지연 (축 2-1)</b><br>'
        '3구간: t1 출동(차량) · t2 들것 도보 · t3 이송(차량)<br>'
        '<b>지연</b> = 경사 반영 시간 − 경사 0 시간<br>'
        '경사계수 b_up = <b>0.489 s/m</b> (네비 실측, p&lt;0.001)<br>'
        '속도 보정 0.34 · 긴급주행 k=0.70<br>'
        '<span style="color:#666">좌측 상단 레이어 버튼으로 지표 전환</span></div>')
    m.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=False).add_to(m)

    os.makedirs("demo", exist_ok=True)
    m.save(OUT)
    print(f"저장: {OUT}")
    print(f"  집계구 {len(g)}개 · 레이어 {len(layers)}개 + 안전센터 + 병원")
    print(f"  총 도달시간 {g.total_min.min():.1f}~{g.total_min.max():.1f}분 "
          f"· 지연 {g.delay_min.min():.2f}~{g.delay_min.max():.2f}분")


if __name__ == "__main__":
    main()
