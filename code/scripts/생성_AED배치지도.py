# -*- coding: utf-8 -*-
"""AED 실좌표 배치 결과 지도.

레이어를 켜고 끄며 "왜 여기에 놓아야 하는가"를 순서대로 보여준다.
  ① 집계구 수요(구급차를 기다리는 동안 방치되는 고위험 인구·분)
  ② 구급차 도달시간 — 골든타임 4분 초과 여부
  ③ 후보 시설 전체(경로당·편의점·마트·공공시설) — 고지대엔 24시간 상업시설이 없다
  ④ 신규 AED 15개소 + 커버 반경 150m
  ⑤ 시설공백 집계구 — 독립 함체가 필요한 곳
  ⑥ 이전 MCLP 후보 → 실좌표 이동선

실행:  python code/scripts/생성_AED배치지도.py
출력:  results/AED배치지도.html
"""
import os
import sys
import glob
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

for _c in [Path.cwd(), *Path.cwd().parents]:
    if (_c / "README.md").exists() and (_c / ".gitignore").exists():
        os.chdir(_c)
        break

import numpy as np
import pandas as pd
import folium
import branca.colormap as cm
import shapefile
from folium import FeatureGroup, CircleMarker, Marker, Icon, DivIcon, PolyLine
from pyproj import Transformer

OUT = "results/AED배치지도.html"
R_COVER = 150
GOLDEN = 4.0


def tract_polygons():
    """집계구 경계를 WGS84 폴리곤으로. (코드 → [[lat,lon],...])"""
    sf = shapefile.Reader(glob.glob("data/**/bnd_oa_*.shp", recursive=True)[0],
                          encoding="utf-8")
    tr = Transformer.from_crs("EPSG:5179", "EPSG:4326", always_xy=True)
    out = {}
    for sr in sf.iterShapeRecords():
        pts, parts = sr.shape.points, list(sr.shape.parts) + [len(sr.shape.points)]
        rings = []
        for i in range(len(parts) - 1):
            ring = pts[parts[i]:parts[i + 1]]
            if len(ring) < 3:
                continue
            xs, ys = zip(*ring)
            lon, lat = tr.transform(xs, ys)
            rings.append([[la, lo] for lo, la in zip(lon, lat)])
        if rings:
            out[str(sr.record["TOT_OA_CD"])] = max(rings, key=len)
    return out


def main():
    dem = pd.read_csv("data/output/집계구별_출동시간_지연.csv", dtype={"집계구코드": str})
    dem["도달_분"] = dem["t1_출동_분"] + dem["t2_접근_들것_분"]
    dem["초과_분"] = (dem["도달_분"] - GOLDEN).clip(lower=0)
    dem["수요"] = dem["75세이상인구_2026"] * dem["초과_분"]

    new = pd.read_csv("data/output/AED_실좌표_후보.csv")
    gap = pd.read_csv("data/output/AED_시설공백_집계구.csv", dtype={"집계구코드": str})
    snap = pd.read_csv("data/output/AED_이전후보_실좌표매핑.csv",
                       dtype={"집계구코드": str})
    pool = pd.read_csv("data/output/AED_후보시설_통합.csv")
    poly = tract_polygons()

    m = folium.Map(location=[35.1255, 129.0415],
                   zoom_start=15, tiles=None, control_scale=True)
    folium.TileLayer("cartodbpositron", name="배경: 밝게").add_to(m)
    folium.TileLayer("OpenStreetMap", name="배경: OSM").add_to(m)

    # ── ① 수요 ──────────────────────────────────────────────────────────────
    g1 = FeatureGroup(name="① 집계구 수요(75세+ × 골든타임 초과분)", show=True)
    vmax = float(dem.수요.quantile(0.95)) or 1.0
    cmap = cm.LinearColormap(["#f7f7f7", "#fdd49e", "#fc8d59", "#d7301f"],
                             vmin=0, vmax=vmax, caption="수요 = 75세+ × (도달시간−4분)")
    for _, r in dem.iterrows():
        p = poly.get(r["집계구코드"])
        if not p:
            continue
        folium.Polygon(p, color="#ffffff", weight=0.6, fill=True,
                       fill_color=cmap(min(r["수요"], vmax)), fill_opacity=0.72,
                       tooltip=(f"{r['행정동']}<br>75세+ {r['75세이상인구_2026']:.0f}명"
                                f"<br>도달 {r['도달_분']:.1f}분"
                                f" (초과 {r['초과_분']:.1f})"
                                f"<br>수요 {r['수요']:.0f}")).add_to(g1)
    g1.add_to(m)
    m.add_child(cmap)

    # ── ② 골든타임 초과 ─────────────────────────────────────────────────────
    g2 = FeatureGroup(name="② 골든타임 4분 초과 집계구", show=False)
    for _, r in dem[dem.초과_분 > 0].iterrows():
        p = poly.get(r["집계구코드"])
        if p:
            folium.Polygon(p, color="#c0392b", weight=1.2, fill=True,
                           fill_color="#c0392b", fill_opacity=0.30,
                           tooltip=(f"{r['행정동']} · 도달 "
                                    f"{r['도달_분']:.1f}분")).add_to(g2)
    g2.add_to(m)

    # ── ③ 후보 시설 전체 ────────────────────────────────────────────────────
    COL = {"경로당": "#8e44ad", "편의점": "#16a085", "마트": "#2980b9",
           "주민센터": "#d35400", "공공시설": "#7f8c8d", "파출소·지구대": "#c0392b",
           "소방서·안전센터": "#e74c3c", "주유소": "#f39c12", "도시철도역": "#34495e"}
    g3 = FeatureGroup(name=f"③ 후보 시설 전체 ({len(pool)})", show=False)
    for r in pool.itertuples():
        CircleMarker([r.lat, r.lon], radius=3,
                     color=COL.get(r.유형, "#95a5a6"), fill=True, fill_opacity=0.75,
                     weight=0.5,
                     tooltip=f"{r.이름}<br>{r.유형} · 야간 {r.야간접근_현재}").add_to(g3)
    g3.add_to(m)

    g3b = FeatureGroup(name=f"③b 경로당만 ({int((pool.유형=='경로당').sum())})", show=False)
    for r in pool[pool.유형 == "경로당"].itertuples():
        CircleMarker([r.lat, r.lon], radius=4, color="#8e44ad", fill=True,
                     fill_opacity=0.9, weight=0.5, tooltip=r.이름).add_to(g3b)
    g3b.add_to(m)

    # ── ④ 신규 AED + 커버 ───────────────────────────────────────────────────
    g4c = FeatureGroup(name="④ 신규 AED 커버 (R=150m)", show=True)
    g4 = FeatureGroup(name=f"④ 신규 AED {len(new)}개소", show=True)
    for r in new.itertuples():
        folium.Circle([r.lat, r.lon], radius=R_COVER, color="#1a9641", weight=1,
                      fill=True, fill_color="#1a9641", fill_opacity=0.12).add_to(g4c)
        now_ok = r.야간접근_현재 >= 1.0
        Marker([r.lat, r.lon],
               icon=DivIcon(html=f'<div style="font-size:15px;color:'
                                 f'{"#1a9641" if now_ok else "#d35400"};'
                                 f'text-shadow:0 0 3px #fff">★</div>'),
               tooltip=(f"<b>{r.순위}. {r.이름}</b><br>{r.유형} · {r.행정동}"
                        f"<br>{'✅ 현재도 야간 가능' if now_ok else '⚠ '+r.필요조치}"
                        f"<br>커버 집계구 {r.커버수요수}")).add_to(g4)
    g4c.add_to(m)
    g4.add_to(m)

    # ── ⑤ 시설공백 ──────────────────────────────────────────────────────────
    # 산지를 포함한 초대형 집계구는 중심점이 무인 산지라 '시설 없음'이 인위적이다.
    # 실제 공백과 구분해서 칠한다(검정=실제, 빗금 회색=산지 아티팩트).
    mnt = gap["산지포함_의심"] if "산지포함_의심" in gap.columns else pd.Series(False, index=gap.index)
    n_real = int((~mnt).sum())
    g5 = FeatureGroup(name=f"⑤ 시설공백 — 실제 {n_real}곳", show=True)
    g5m = FeatureGroup(name=f"⑤b 시설공백 — 산지 아티팩트 {int(mnt.sum())}곳", show=True)
    for i, r in gap.iterrows():
        p = poly.get(r["집계구코드"])
        if not p:
            continue
        is_mnt = bool(mnt.loc[i])
        folium.Polygon(
            p, color="#7f8c8d" if is_mnt else "#000000",
            weight=2.0, fill=True,
            fill_color="#bdc3c7" if is_mnt else "#000000",
            fill_opacity=0.25 if is_mnt else 0.45,
            tooltip=((f"<b>⚠ 산지 포함 집계구</b><br>{r['행정동']}"
                      f"<br>면적 {r['면적_ha']:.0f}ha (중앙값 2.5ha)"
                      f"<br>중심점이 무인 산지에 있어 도달 {r['도달_분']:.1f}분·"
                      f"시설공백이 과대평가됨<br>실거주지는 도로변에 분포")
                     if is_mnt else
                     (f"<b>독립 함체 필요</b><br>{r['행정동']}"
                      f"<br>75세+ {r['75세이상인구_2026']:.0f}명"
                      f" · 도달 {r['도달_분']:.1f}분"))
        ).add_to(g5m if is_mnt else g5)
    g5.add_to(m)
    g5m.add_to(m)

    # ── ⑥ 이전 후보 → 실좌표 이동선 ─────────────────────────────────────────
    g6 = FeatureGroup(name="⑥ 이전 MCLP 후보 → 실좌표 이동", show=False)
    prev = pd.read_csv("data/output/AED_신규후보_15.csv")
    prev.columns = [c.strip().lstrip("﻿") for c in prev.columns]
    prev["TOT_OA_CD"] = prev["TOT_OA_CD"].astype(str)
    pv = prev.set_index("TOT_OA_CD")
    for r in snap.itertuples():
        if r.집계구코드 not in pv.index or pd.isna(r.거리_m):
            continue
        o = pv.loc[r.집계구코드]
        CircleMarker([o.lat, o.lon], radius=3, color="#7f8c8d", fill=True,
                     tooltip="이전 후보(집계구 중심)").add_to(g6)
        PolyLine([[o.lat, o.lon], [r.lat, r.lon]], color="#7f8c8d",
                 weight=1.5, dash_array="4").add_to(g6)
    g6.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    n_now = int((new.야간접근_현재 >= 1.0).sum())
    n_real = int((~mnt).sum())
    legend = f"""
<div style="position:fixed;top:10px;left:50px;z-index:9999;background:rgba(255,255,255,.95);
 padding:10px 13px;border-radius:6px;font-family:-apple-system,'Apple SD Gothic Neo',sans-serif;
 font-size:12px;max-width:330px;box-shadow:0 1px 6px rgba(0,0,0,.3)">
<b style="font-size:14px">산복도로 AED 실좌표 배치</b><br>
<span style="color:#555">부산 동구 초량·좌천동 · 집계구 78</span><hr style="margin:6px 0">
<b>신규 {len(new)}개소</b>로 위험가중 수요 <b>69.0%</b> 커버<br>
&nbsp;<span style="color:#1a9641">★</span> 현재도 야간 가능 {n_now} &nbsp;
<span style="color:#d35400">★</span> 외벽 함체 필요 {len(new)-n_now}<br>
&nbsp;<span style="color:#000">■</span> 독립 함체 신설 {n_real}곳
&nbsp;<span style="color:#bdc3c7">■</span> 산지 아티팩트 {len(gap)-n_real}곳<hr style="margin:6px 0">
<span style="color:#8e44ad">●</span> 경로당 {int((pool.유형=='경로당').sum())}
&nbsp;<span style="color:#16a085">●</span> 편의점
&nbsp;<span style="color:#2980b9">●</span> 마트<br>
<span style="color:#666">고지대엔 24시간 상업시설이 없어 경로당이 유일한 앵커다.</span>
</div>"""
    m.get_root().html.add_child(folium.Element(legend))

    os.makedirs("results", exist_ok=True)
    m.save(OUT)
    print(f"저장: {OUT}")
    print(f"  집계구 {len(dem)} · 후보시설 {len(pool)} · 신규 AED {len(new)} · 공백 {len(gap)}")


if __name__ == "__main__":
    main()
