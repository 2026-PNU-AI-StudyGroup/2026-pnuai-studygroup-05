# -*- coding: utf-8 -*-
"""AED 실좌표 배치 — 집계구 중심이 아니라 '실제 설치 가능한 지점'을 고른다.

기존 MCLP(노트북 11)는 집계구 중심 좌표를 후보로 썼다. 그 좌표는 행정 경계의
기하학적 중심일 뿐이라 실제로는 산비탈 한가운데이거나 사유지일 수 있어
"여기에 설치하라"고 말할 수 없다. 이 스크립트는 후보 집합을 실제 시설로 바꾼다.

세 가지 요구를 반영한다.
  1) 접근성   — 편의점·마트·공공시설 등 사람이 드나드는 곳.
                야간에 문을 닫으면 심야 심정지에 무용하므로 야간 접근을 등급화.
  2) 출동지연 — 구급차가 늦게 닿는 집계구일수록 수요 가중을 높인다.
                (t1 출동 + t2 들것 = 환자에게 닿는 시간)
  3) 이전 후보 — 노트북 11 이 낸 AED 신규후보 15개를 존중한다. 그 결과는
                 설치된 AED 106개의 야간 커버를 이미 빼고 남은 고위험 집계구라
                 재산출하지 않고 그대로 이어받아 실좌표로만 바꾼다.
                 (설치된 AED 좌표가 있으면 중복 배제를 한 번 더 적용)

실행:  python code/scripts/AED_실좌표_배치.py
출력:  data/output/AED_실좌표_후보.csv
       data/output/AED_실좌표_후보_전체POI.csv   (후보 전수 + 점수)
"""
import os
import sys
import json
import glob
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

# ── 저장소 루트로 이동: 실행 위치와 무관하게 data/ 상대경로 유지 ──
for _c in [Path.cwd(), *Path.cwd().parents]:
    if (_c / "README.md").exists() and (_c / ".gitignore").exists():
        os.chdir(_c)
        break

import numpy as np
import pandas as pd
import requests
import shapefile
from pyproj import Transformer
from scipy.spatial import cKDTree

# ============================================================================
# 파라미터
# ============================================================================
R_COVER      = 150.0   # AED 유효 커버 반경(m). 노트북 11과 동일 기준.
N_NEW        = 15      # 신규 배치 수. 기존 MCLP 결과와 비교 가능하도록 맞춤.
GOLDEN_MIN   = 4.0     # 골든타임(분). 이 시간을 넘는 초과분만 AED 가 메울 공백으로 본다.
UNCOV_BOOST  = 2.0     # 야간 미커버 확정 집계구 가중(기존 AED 좌표 미확보 시)
SNAP_MAX_M   = 250.0   # 집계구 중심에서 이 거리 안의 시설만 그 집계구의 후보로 인정
CACHE        = "outputs/poi_overpass.json"   # Overpass 응답 캐시(중간 산출물)

OUT_MAIN = "data/output/AED_실좌표_후보.csv"
OUT_ALL  = "data/output/AED_실좌표_후보_전체POI.csv"

BBOX = (35.1100, 129.0260, 35.1400, 129.0580)   # S, W, N, E (대상지 + 버퍼)

UA = {"User-Agent": "PNU-AED-Study/1.0 (academic research)",
      "Accept": "application/json"}

# ── 야간 접근 등급 ───────────────────────────────────────────────────────────
# AED 는 '심정지가 목격된 순간 곧바로 꺼내 쓸 수 있어야' 의미가 있다.
# 야간에 잠기는 건물 안의 AED 는 심야 발생에 대해 사실상 없는 것과 같다.
NIGHT_24H = 1.00   # 24시간 개방
NIGHT_LATE = 0.40  # 심야 이전까지(대개 22~24시) — 부분 커버
NIGHT_DAY = 0.10   # 주간 전용 — 외벽 설치 시에만 야간 가능하므로 낮게

# 한국 24시간 편의점 브랜드. OSM opening_hours 태그가 대부분 비어 있어
# 브랜드명으로 판정한다(실측: 22건 중 24/7 태그는 3건뿐).
CVS_BRANDS = ("cu", "gs25", "seven", "세븐일레븐", "이마트24", "emart24",
              "미니스톱", "ministop", "storyway", "24시")
# shop=convenience 로 잘못 태깅된 것들(세탁소·문구점 등)을 걸러낸다.
NOT_CVS = ("세탁", "방구", "문구", "부동산", "미용")


def classify(tags):
    """OSM 태그 → (유형, 야간접근_현재, 야간접근_조치후, 필요조치). 후보가 아니면 None.

    야간접근을 두 값으로 나눈 이유:
      산복도로 고지대에는 24시간 상업시설이 거의 없다. 그래서 '지금 당장 야간에
      쓸 수 있는 곳'만 후보로 두면 고지대는 영원히 배치 대상이 되지 못한다.
      주민센터·경로당은 실내라 야간에 잠기지만, 외벽에 24시간 개방 함체를 달면
      바로 24시간이 된다(비용이 낮고 지자체 단독으로 집행 가능).
      따라서 최적화는 '조치 후' 기준으로 하고, 어떤 조치가 필요한지 함께 낸다.
    """
    name = (tags.get("name") or tags.get("name:ko") or "").strip()
    low = name.lower()
    shop = tags.get("shop")
    amen = tags.get("amenity")
    oh = (tags.get("opening_hours") or "").strip()
    is_24h = oh in ("24/7", "Mo-Su 00:00-24:00")

    # ── 의료기관 제외 ──
    # OSM 한국 데이터는 한의원·요양병원·의원을 전부 amenity=hospital 로 태깅한다
    # (실측: hospital 13건 중 종합병원은 소수). 게다가 의료기관은 자체 제세동
    # 장비를 갖추므로 공용 AED 의 한계효용이 낮다. 후보에서 뺀다.
    if amen in ("hospital", "clinic") or any(
            k in name for k in ("병원", "의원", "한의", "치과", "요양")):
        return None

    if shop == "convenience":
        if any(k in name for k in NOT_CVS):
            return None                                   # 세탁편의점 등 오분류
        if is_24h or any(b in low for b in CVS_BRANDS):
            return ("편의점", 1.0, 1.0, "없음(이미 24시간)")
        return ("편의점", 0.4, 0.9, "운영시간 확인 후 야간 개방 협의")

    if shop in ("supermarket", "mall", "department_store", "greengrocer"):
        return ("마트", 1.0 if is_24h else 0.4, 0.6, "영업시간 외 외벽 함체 설치")

    if amen == "police":
        return ("파출소·지구대", 1.0, 1.0, "없음(24시간 상주)")
    if amen == "fire_station":
        return ("소방서·안전센터", 1.0, 1.0, "없음(24시간 상주)")
    if amen == "fuel":
        return ("주유소", 1.0 if is_24h else 0.4, 0.8, "야간 무인 구역에 함체 설치")

    # ── 공공시설: 산복도로 고지대의 사실상 유일한 후보 ──
    if amen in ("townhall", "community_centre", "library", "post_office",
                "social_facility") or tags.get("office") == "government":
        if "경로당" in name or "노인" in name:
            return ("경로당", 0.1, 1.0, "★ 외벽 24시간 함체 설치")
        if "주민센터" in name or amen == "townhall":
            return ("주민센터", 0.1, 1.0, "★ 외벽 24시간 함체 설치")
        return ("공공시설", 0.1, 1.0, "★ 외벽 24시간 함체 설치")

    if tags.get("railway") == "station":
        return ("도시철도역", 0.4, 0.7, "역사 외부 개방구역에 설치")

    return None      # 학교·은행·종교시설 등 제외


# ============================================================================
# 1. 집계구 좌표 + 출동지연
# ============================================================================
def load_tracts():
    shp = glob.glob("data/**/bnd_oa_*.shp", recursive=True)
    if not shp:
        raise SystemExit("집계구 경계 SHP 를 찾을 수 없습니다. data/input/ 배치를 확인하세요.")
    sf = shapefile.Reader(shp[0], encoding="utf-8")
    tr = Transformer.from_crs("EPSG:5179", "EPSG:4326", always_xy=True)

    def centroid(pts, parts):
        """가장 큰 외곽 링의 면적가중 무게중심."""
        idx = list(parts) + [len(pts)]
        best, bestA = None, 0.0
        for i in range(len(idx) - 1):
            ring = np.asarray(pts[idx[i]:idx[i + 1]], dtype=float)
            if len(ring) < 3:
                continue
            x, y = ring[:, 0], ring[:, 1]
            a = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
            if abs(a) > abs(bestA):
                bestA, best = a, ring
        ring = best if best is not None else np.asarray(pts, dtype=float)
        x, y = ring[:, 0], ring[:, 1]
        cr = x * np.roll(y, -1) - np.roll(x, -1) * y
        A = 0.5 * cr.sum()
        if abs(A) < 1e-9:
            return x.mean(), y.mean()
        return (((x + np.roll(x, -1)) * cr).sum() / (6 * A),
                ((y + np.roll(y, -1)) * cr).sum() / (6 * A))

    rows = []
    for sr in sf.iterShapeRecords():
        cx, cy = centroid(sr.shape.points, sr.shape.parts)
        lon, lat = tr.transform(cx, cy)
        rows.append((str(sr.record["TOT_OA_CD"]), lon, lat))
    return pd.DataFrame(rows, columns=["집계구코드", "oa_lon", "oa_lat"])


def load_demand():
    d = pd.read_csv("data/output/집계구별_출동시간_지연.csv", dtype={"집계구코드": str})
    oa = load_tracts()
    m = d.merge(oa, on="집계구코드", how="left")
    if m.oa_lon.isna().any():
        raise SystemExit(f"좌표 조인 실패 {int(m.oa_lon.isna().sum())}건")
    # 요구 2: 구급차가 환자에게 닿는 시간 = t1(출동) + t2(들것 접근)
    m["도달_분"] = m["t1_출동_분"] + m["t2_접근_들것_분"]
    return m


# ============================================================================
# 2. POI 수집 (Overpass)
# ============================================================================
def fetch_poi():
    if os.path.exists(CACHE):
        print(f"[POI] 캐시 사용: {CACHE}")
        return json.load(open(CACHE, encoding="utf-8"))["elements"]

    S, W, N, E = BBOX
    q = f"""[out:json][timeout:120];
(
 nwr["shop"~"^(convenience|supermarket|mall|department_store|greengrocer)$"]({S},{W},{N},{E});
 nwr["amenity"~"^(police|fire_station|community_centre|townhall|library|social_facility|pharmacy|hospital|clinic|post_office|fuel)$"]({S},{W},{N},{E});
 nwr["office"="government"]({S},{W},{N},{E});
 nwr["emergency"="defibrillator"]({S},{W},{N},{E});
 nwr["railway"="station"]({S},{W},{N},{E});
);
out tags center;"""
    last = None
    for url in ("https://overpass-api.de/api/interpreter",
                "https://overpass.kumi.systems/api/interpreter"):
        try:
            r = requests.post(url, data={"data": q}, headers=UA, timeout=180)
            r.raise_for_status()
            js = r.json()
            os.makedirs(os.path.dirname(CACHE), exist_ok=True)
            json.dump(js, open(CACHE, "w", encoding="utf-8"), ensure_ascii=False)
            print(f"[POI] {url.split('/')[2]} 에서 {len(js['elements'])}건 수집")
            return js["elements"]
        except Exception as e:                      # noqa: BLE001
            last = e
            time.sleep(2)
    raise SystemExit(f"Overpass 접속 실패: {last}")


def poi_frame(elements):
    rows, aed_osm = [], []
    for e in elements:
        t = e.get("tags", {})
        lon = e.get("lon") or (e.get("center") or {}).get("lon")
        lat = e.get("lat") or (e.get("center") or {}).get("lat")
        if lon is None or lat is None:
            continue
        if t.get("emergency") == "defibrillator":
            aed_osm.append((lon, lat))
            continue
        c = classify(t)
        if c is None:
            continue
        kind, now, after, act = c
        rows.append(dict(이름=(t.get("name") or t.get("name:ko") or "(무명)"),
                         유형=kind, 야간접근_현재=now, 야간접근_조치후=after,
                         필요조치=act, lon=lon, lat=lat,
                         운영시간=t.get("opening_hours", "")))
    return pd.DataFrame(rows), aed_osm


# ============================================================================
# 3. 기존 AED (요구 3)
# ============================================================================
def load_existing_aed(aed_osm):
    """설치된 AED 좌표(선택). 없으면 이전 MCLP 후보 15개를 기준으로 쓴다."""
    # 설치된 AED 좌표가 있으면 중복 배제를 추가로 적용한다(선택).
    for p in ("outputs/aed_donggu.csv", "data/input/aed_donggu.csv"):
        if os.path.exists(p):
            a = pd.read_csv(p)
            latc = next((c for c in a.columns if "Lat" in c or "lat" in c), None)
            lonc = next((c for c in a.columns if "Lon" in c or "lon" in c), None)
            if latc and lonc:
                a = a.dropna(subset=[latc, lonc])
                print(f"[기존AED] {p} 에서 {len(a)}건")
                return list(zip(a[lonc].astype(float), a[latc].astype(float))), "공공데이터포털"
    if aed_osm:
        print(f"[기존AED] OSM 에서 {len(aed_osm)}건")
        return aed_osm, "OSM"
    print("[이전후보] 노트북 11 의 신규후보 15개를 기준으로 사용"
          " (설치 AED 106개의 야간 커버가 이미 반영된 결과)")
    return [], "이전후보(노트북11)"


# ============================================================================
# 4. MCLP (그리디) — 후보는 실제 시설
# ============================================================================
def main():
    dem = load_demand()
    poi, aed_osm = poi_frame(fetch_poi())
    aed_xy, aed_src = load_existing_aed(aed_osm)

    # 미터 좌표계로 변환(거리 계산용)
    to_m = Transformer.from_crs("EPSG:4326", "EPSG:5186", always_xy=True)
    dem["x"], dem["y"] = to_m.transform(dem.oa_lon.values, dem.oa_lat.values)
    poi["x"], poi["y"] = to_m.transform(poi.lon.values, poi.lat.values)

    # ── 요구 3: 기존 AED 가 야간 커버하는 집계구는 수요에서 제외 ──
    if aed_xy:
        ax, ay = to_m.transform([p[0] for p in aed_xy], [p[1] for p in aed_xy])
        d_aed = cKDTree(np.c_[ax, ay]).query(np.c_[dem.x, dem.y])[0]
        dem["기존AED_거리_m"] = d_aed
        dem["기존커버"] = d_aed <= R_COVER
    else:
        # 이전 MCLP 후보 15개를 '확실히 AED 가 필요한 곳'으로 보고 가중을 올린다.
        # 그 15개는 설치 AED 106개의 야간 커버를 이미 제외한 결과이므로,
        # 여기에 가중을 주는 것이 곧 기존 배치를 존중하는 것이다.
        # 나머지 63개도 제외하지 않고 남긴다 — 전수 제외는 수요를 과소평가한다.
        prev = pd.read_csv("data/output/AED_신규후보_15.csv")
        prev.columns = [c.strip().lstrip("\ufeff") for c in prev.columns]
        uncov = set(prev["TOT_OA_CD"].astype(str))
        dem["기존AED_거리_m"] = np.nan
        dem["기존커버"] = False
        dem["미커버확정"] = dem["집계구코드"].isin(uncov)

    # ── 요구 2: 출동지연 가중 수요 ──
    # 수요 = 75세이상인구 × (구급차 도달시간 − 골든타임)
    #   = "구급차를 기다리는 동안 방치되는 고위험 인구·분"
    # 골든타임 안에 닿는 집계구는 AED 없이도 구급차가 감당하므로 0 이 된다.
    dem["초과_분"] = (dem["도달_분"] - GOLDEN_MIN).clip(lower=0)
    dem["수요"] = dem["75세이상인구_2026"] * dem["초과_분"]
    if "미커버확정" in dem:
        dem.loc[dem["미커버확정"], "수요"] *= UNCOV_BOOST
    dem.loc[dem["기존커버"], "수요"] = 0.0
    live = dem[dem.수요 > 0].copy()

    # ── 후보 × 수요 커버 행렬 ──
    tree = cKDTree(np.c_[live.x, live.y])
    cover = [tree.query_ball_point((r.x, r.y), R_COVER) for r in poi.itertuples()]
    poi["커버수요수"] = [len(c) for c in cover]

    # 집계구에서 너무 먼(=주민 생활동선 밖) 시설은 후보에서 제외
    dmin = cKDTree(np.c_[dem.x, dem.y]).query(np.c_[poi.x, poi.y])[0]
    poi["최근접집계구_m"] = dmin
    valid = (dmin <= SNAP_MAX_M)

    w = live["수요"].to_numpy()
    chosen, remaining = [], w.copy()
    for _ in range(N_NEW):
        best, bestgain = -1, 0.0
        for i, idxs in enumerate(cover):
            if not valid[i] or i in chosen or not idxs:
                continue
            # 요구 1: 야간에 못 여는 시설은 그만큼 이득을 깎는다
            gain = remaining[idxs].sum() * poi.야간접근_조치후.iat[i]
            if gain > bestgain:
                best, bestgain = i, gain
        if best < 0:
            break
        chosen.append(best)
        remaining[cover[best]] = 0.0

    total = w.sum()
    covered = total - remaining.sum()

    res = poi.iloc[chosen].copy()
    res.insert(0, "순위", range(1, len(res) + 1))
    res["커버수요"] = [round(w[cover[i]].sum(), 3) for i in chosen]
    res["행정동"] = [live.iloc[cover[i]].행정동.mode().iat[0] if cover[i] else ""
                   for i in chosen]

    # ── 시설 공백 분석 ──────────────────────────────────────────────────────
    # 반경 안에 후보 시설이 아예 없는 집계구는 '기존 시설 활용'으로는 손댈 수 없다.
    # 독립 함체(가로등·전신주 부착형 등)를 새로 세워야 하므로 따로 보고한다.
    ptree = cKDTree(np.c_[poi.x[valid], poi.y[valid]])
    n_near = [len(ptree.query_ball_point((r.x, r.y), R_COVER))
              for r in live.itertuples()]
    live = live.assign(반경내_시설수=n_near)
    gap = live[live.반경내_시설수 == 0].copy()
    gap_share = gap.수요.sum() / live.수요.sum() * 100 if len(live) else 0.0
    gap_out = gap.sort_values("수요", ascending=False)[
        ["집계구코드", "행정동", "75세이상인구_2026", "t1_출동_분",
         "t2_접근_들것_분", "도달_분", "초과_분", "수요", "oa_lon", "oa_lat"]]
    gap_out.to_csv("data/output/AED_시설공백_집계구.csv", index=False,
                   encoding="utf-8-sig")

    # ── 이전 MCLP 15개 집계구 → 실제 설치 가능 시설 매핑 ────────────────────
    # 노트북 11 의 결과(AED_신규후보_15.csv)는 기존 설치 AED 106개의 야간 커버를
    # 이미 제외하고 남은 고위험 집계구다. 즉 '기존 AED 배치를 존중한' 결과이므로
    # 그대로 살리되, 집계구 중심이라는 한계만 실좌표로 바꿔준다.
    prev = pd.read_csv("data/output/AED_신규후보_15.csv")
    prev.columns = [c.strip().lstrip("\ufeff") for c in prev.columns]
    prev["TOT_OA_CD"] = prev["TOT_OA_CD"].astype(str)
    px, py = to_m.transform(prev.lon.values, prev.lat.values)
    vpoi = poi[valid].reset_index(drop=True)
    vt = cKDTree(np.c_[vpoi.x, vpoi.y])
    snap = []
    for i, r in enumerate(prev.itertuples()):
        idxs = vt.query_ball_point((px[i], py[i]), R_COVER)
        if idxs:
            # 반경 안에서 야간접근(조치 후)이 가장 좋고, 그다음 가까운 시설
            cand = vpoi.iloc[idxs].copy()
            cand["거리_m"] = np.hypot(cand.x - px[i], cand.y - py[i])
            cand = cand.sort_values(["야간접근_조치후", "거리_m"],
                                    ascending=[False, True])
            b = cand.iloc[0]
            snap.append(dict(순위=r.rank, 행정동=r.dong, 집계구코드=r.TOT_OA_CD,
                             위험도=r.risk_norm, 설치시설=b.이름, 시설유형=b.유형,
                             야간접근_현재=b.야간접근_현재, 필요조치=b.필요조치,
                             거리_m=round(b.거리_m, 1), lon=b.lon, lat=b.lat))
        else:
            snap.append(dict(순위=r.rank, 행정동=r.dong, 집계구코드=r.TOT_OA_CD,
                             위험도=r.risk_norm, 설치시설="(반경 내 시설 없음)",
                             시설유형="—", 야간접근_현재=np.nan,
                             필요조치="★ 독립 함체 신설", 거리_m=np.nan,
                             lon=r.lon, lat=r.lat))
    snap = pd.DataFrame(snap)
    snap.to_csv("data/output/AED_이전후보_실좌표매핑.csv", index=False,
                encoding="utf-8-sig")

    os.makedirs("data/output", exist_ok=True)
    cols = ["순위", "이름", "유형", "행정동", "야간접근_현재", "야간접근_조치후",
            "필요조치", "커버수요", "커버수요수", "최근접집계구_m", "lon", "lat"]
    res[cols].to_csv(OUT_MAIN, index=False, encoding="utf-8-sig")
    poi.assign(선정=poi.index.isin(chosen)).to_csv(OUT_ALL, index=False,
                                                 encoding="utf-8-sig")

    # ── 리포트 ──
    print("\n" + "=" * 74)
    print(f"후보 시설 {len(poi)}개(유효 {int(valid.sum())}) · 수요 집계구 "
          f"{len(live)}/{len(dem)} · 기존AED 출처: {aed_src}")
    print(f"신규 {len(chosen)}개소로 위험가중 수요 {covered/total*100:.1f}% 커버")
    print("=" * 74)
    print(res[["순위", "이름", "유형", "행정동", "야간접근_현재", "필요조치",
               "커버수요수"]].to_string(index=False))
    print(f"\n유형 분포:\n{res.유형.value_counts().to_string()}")
    print("\n" + "=" * 74)
    print(f"[시설 공백] 반경 {R_COVER:.0f}m 안에 후보 시설이 하나도 없는 집계구 "
          f"{len(gap)}개 — 전체 수요의 {gap_share:.1f}%")
    print("  기존 시설로는 커버 불가 → 독립 함체(가로등·전신주 부착형) 신설 대상")
    print("=" * 74)
    if len(gap):
        g = gap_out.head(10).copy()
        g["도달_분"] = g["도달_분"].round(1); g["수요"] = g["수요"].round(0)
        print(g[["행정동", "75세이상인구_2026", "도달_분", "수요"]].to_string(index=False))
        print(f"\n  동별 공백: {gap.행정동.value_counts().to_dict()}")

    print("\n" + "=" * 74)
    n_ok = int((snap.설치시설 != "(반경 내 시설 없음)").sum())
    print(f"[이전 MCLP 15개 → 실좌표] 시설 매핑 {n_ok}/15 · 독립 함체 필요 {15-n_ok}")
    print("  (이전 결과는 기존 설치 AED 106개의 야간 커버를 이미 제외한 것)")
    print("=" * 74)
    print(snap[["순위", "행정동", "설치시설", "시설유형", "필요조치",
                "거리_m"]].to_string(index=False))
    hit = snap[snap.설치시설 != "(반경 내 시설 없음)"]
    uniq = hit.설치시설.nunique()
    print(f"\n  실제 설치 지점 {uniq}개소로 {len(hit)}개 집계구 커버"
          f"(같은 시설이 인접 집계구를 함께 덮음)")
    need = snap[snap.설치시설 == "(반경 내 시설 없음)"]
    if len(need):
        print(f"  독립 함체 신설 {len(need)}개소 — 동별 "
              f"{need.행정동.value_counts().to_dict()}")
    print(f"\n저장: {OUT_MAIN}\n      {OUT_ALL}\n      data/output/AED_시설공백_집계구.csv\n      data/output/AED_이전후보_실좌표매핑.csv")
    return res


if __name__ == "__main__":
    main()
