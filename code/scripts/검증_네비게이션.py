# -*- coding: utf-8 -*-
"""네비게이션 API 검증 — 경사계수를 실주행 데이터에서 추정

docs/02 의 경사비용은 β=6 을 가정값으로 쓴다(민감도 4/6/8). 8/10 회의에서 이 부분이
"임의로 계산"이라 지적되어, 계수를 실제 주행 소요시간에서 추정하도록 만든 스크립트다.

  회귀:  소요시간 ~ 도로거리 + 누적오르막 + 누적내리막
  → 누적오르막 계수 b_up = "1m 오를 때 추가되는 초"  ← `출동시간_추정.py` 가 쓸 값

핵심 설계 — 왕복 자연실험
  같은 구간을 양방향으로 모두 수집한다.
    A_out  안전센터 → 집계구   출동 (표고차 중앙 +14.9m)
    A_back 집계구 → 안전센터   같은 길 역방향 (중앙 −14.9m)
    B      집계구 → 병원       이송
  거리·도로·교통량이 거의 동일하고 경사 부호만 반대이므로, 두 소요시간의 차이는
  사실상 순수한 경사 효과다. 교란변수가 최소화된다.

왜 표고를 경로에서 다시 뽑나
  필요한 값은 출발–도착 직선 표고차가 아니라 실제 주행 경로의 누적 오르막이다.
  (직선 표고차 100m 라도 산을 돌아 180m 올랐다 80m 내려오는 경로일 수 있다)
  별도 DEM 을 새로 받지 않고 그래프 노드 표고 4,756점을 보간해 쓴다.
  OpenTopoData 공개 API 는 일 1,000요청 제한이라 1,326개 경로에는 부족하다.

왜 119 출동기록(축5)이 아니라 네비인가
  부산 119 소방출동정보 API(getTodayInfo)의 출력 항목은
  regtime·dsraddr·dsrkndcd·dsrclscd·dsrsizecd·juriswardid1·2 뿐이다.
  출동시각·현장도착시각·좌표가 없어 소요시간 검증에 쓸 수 없다.
  `검증_119출동.py` 가 이 API 를 수요(출동 건수) 검증에 쓰는 것이 정확한 용법이며,
  도달시간 검증 경로는 네비게이션 API 가 유일하다.

사용법
  1) NCP 콘솔 → Application Services → Maps → Application 등록 → Directions 5 선택
  2) 저장소 루트 .env 에 아래 두 줄 추가 (다른 키와 같은 방식, git 제외됨)
       NCP_KEY_ID=발급받은 Client ID
       NCP_KEY=발급받은 Client Secret
  3) python code/scripts/검증_네비게이션.py
  중단되어도 다시 실행하면 outputs/nav_raw.jsonl 에서 이어받는다.

출력: outputs/nav_calib.csv (+ 중간자료 outputs/od_pairs.csv, outputs/nav_raw.jsonl)
"""
import json
import os
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

# ── 저장소 루트로 이동 ──
for _c in [Path.cwd(), *Path.cwd().parents]:
    if (_c / "README.md").exists() and (_c / ".gitignore").exists():
        os.chdir(_c)
        break

import numpy as np
import pandas as pd
import osmnx as ox
from pyproj import Transformer
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

# Application Services > Maps 의 엔드포인트.
# 구버전 AI·NAVER API 는 naveropenapi.apigw.ntruss.com 을 쓰는데, 신규 Maps 상품에서
# 발급한 키로 구버전 주소를 호출하면 210 Permission Denied
# ("A subscription to the API is required") 가 반환된다. 도메인이 다르다.
URL = "https://maps.apigw.ntruss.com/map-direction/v1/driving"
URL_LEGACY = "https://naveropenapi.apigw.ntruss.com/map-direction/v1/driving"
OPTION = "trafast"          # traoptimal / tracomfort 로 바꿔 비교 가능
# goal 에 콜론으로 여러 개를 넣어도 응답 경로는 1건만 온다(실측 확인).
# 목적지별 결과가 필요하므로 1건씩 호출한다. 1,326회로 일 한도(150,000)에는 여유가 크다.
BATCH = 1
SLEEP = 0.2

GRAPH = "outputs/graph_drive_conn.graphml"
OA = "outputs/oa_risk.parquet"
SC = "outputs/safety_centers.csv"
OD = "outputs/od_pairs.csv"
RAW = "outputs/nav_raw.jsonl"
OUT = "outputs/nav_calib.csv"

# 축3 확정본 (응급의료법 제2조제5호 권역·지역응급의료센터)
HOSPITALS = [
    ("동아대학교병원", 129.017604, 35.120006),
    ("부산대학교병원", 129.019222, 35.101054),
    ("인제대학교부산백병원", 129.020572, 35.146454),
]

# 네이버 Directions 5 는 "경도,위도" 순서를 요구한다. always_xy=True 로 순서를 확정한다.
TO_WGS = Transformer.from_crs("EPSG:5186", "EPSG:4326", always_xy=True)
TO_M = Transformer.from_crs("EPSG:4326", "EPSG:5186", always_xy=True)


def _sf(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def load_graph():
    return ox.load_graphml(GRAPH, edge_dtypes={"length": float},
                           node_dtypes={"elev": _sf})


def build_elev_interp(G):
    """노드 표고(EPSG:5186) → 임의 좌표의 표고. `출동시간_추정.py` 와 동일 방식."""
    pts, vals = [], []
    for _, d in G.nodes(data=True):
        if d.get("elev") == d.get("elev") and d.get("elev") is not None:
            pts.append((float(d["x"]), float(d["y"])))
            vals.append(float(d["elev"]))
    pts, vals = np.array(pts), np.array(vals)
    lin, nea = LinearNDInterpolator(pts, vals), NearestNDInterpolator(pts, vals)

    def f(x, y):
        x, y = np.atleast_1d(x), np.atleast_1d(y)
        z = lin(x, y)
        m = np.isnan(z)
        if m.any():
            z[m] = nea(x[m], y[m])
        return z
    return f


# ============================================================================
# 1. OD 쌍 설계
# ============================================================================
def build_od():
    G = load_graph()
    ef = build_elev_interp(G)

    oa = pd.read_parquet(OA).drop(columns=["geometry"])
    oa["lon"], oa["lat"] = TO_WGS.transform(oa.cx.values, oa.cy.values)
    oa["elev"] = ef(oa.cx.values, oa.cy.values)

    sc = pd.read_csv(SC, encoding="utf-8-sig")
    sc.columns = ["name", "lon", "lat", "dist_m", "region",
                  "osm_name", "fire_dept", "amb_count", "verify"]
    hp = pd.DataFrame(HOSPITALS, columns=["name", "lon", "lat"])

    # 안전센터·병원 표고도 같은 보간기로 뽑아 집계구와 기준을 통일한다.
    # (station 노드의 snap_ref 는 전부 '119안전센터'로 동일해 이름 매칭이 불가능하다)
    for df in (sc, hp):
        x, y = TO_M.transform(df.lon.values, df.lat.values)
        df["elev"] = ef(x, y)

    rows = []

    def add(leg, s, sid, g, gid):
        rows.append(dict(leg=leg, start_id=sid, start_lon=s.lon, start_lat=s.lat,
                         start_elev=s.elev, goal_id=gid, goal_lon=g.lon,
                         goal_lat=g.lat, goal_elev=g.elev, dh=g.elev - s.elev))

    for s in sc.itertuples():                                  # A_out 출동
        for o in oa.itertuples():
            add("A_out", s, s.name, o, o.TOT_OA_CD)
    for o in oa.itertuples():                                  # A_back 역방향
        for s in sc.itertuples():
            add("A_back", o, o.TOT_OA_CD, s, s.name)
    for o in oa.itertuples():                                  # B 이송
        for h in hp.itertuples():
            add("B", o, o.TOT_OA_CD, h, h.name)

    od = pd.DataFrame(rows)
    od.to_csv(OD, index=False, encoding="utf-8-sig")
    calls = sum(-(-n // BATCH) for _, n in od.groupby(["leg", "start_id"]).size().items())
    print(f"OD {len(od)}쌍 → {OD}   (예상 API 호출 {calls}회, 일 한도 150,000)")
    print(od.groupby("leg").dh.describe()[["min", "50%", "max"]].round(1).to_string())
    return od


# ============================================================================
# 2. 수집
# ============================================================================
def read_key(name):
    """.env 우선, 없으면 환경변수. 다른 스크립트의 인증키 관리 방식과 동일하게 맞춘다."""
    if Path(".env").exists():
        for line in open(".env", encoding="utf-8"):
            line = line.strip()
            if line.startswith(name + "="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return os.environ.get(name)


def collect():
    import requests

    kid, key = read_key("NCP_KEY_ID"), read_key("NCP_KEY")
    if not kid or not key:
        raise SystemExit(
            "NCP_KEY_ID / NCP_KEY 를 찾을 수 없습니다.\n"
            "  저장소 루트 .env 에 아래 두 줄을 추가하세요 (git 제외됨):\n"
            "    NCP_KEY_ID=발급받은 Client ID\n"
            "    NCP_KEY=발급받은 Client Secret")
    H = {"x-ncp-apigw-api-key-id": kid, "x-ncp-apigw-api-key": key}
    od = pd.read_csv(OD, encoding="utf-8-sig")

    done = set()
    if Path(RAW).exists():
        for line in open(RAW, encoding="utf-8"):
            r = json.loads(line)
            done.add((r["leg"], str(r["start_id"]), str(r["goal_id"])))
        print(f"이미 수집 {len(done)}건 → 건너뜀")

    f = open(RAW, "a", encoding="utf-8")
    ok = fail = 0
    for (leg, sid), grp in od.groupby(["leg", "start_id"], sort=False):
        grp = grp[[(leg, str(sid), str(g)) not in done for g in grp.goal_id]]
        if grp.empty:
            continue
        s = grp.iloc[0]
        for i in range(0, len(grp), BATCH):
            ch = grp.iloc[i:i + BATCH]
            params = {"start": f"{s.start_lon},{s.start_lat}",
                      "goal": ":".join(f"{r.goal_lon},{r.goal_lat}" for r in ch.itertuples()),
                      "option": OPTION}
            try:
                resp = requests.get(URL, headers=H, params=params, timeout=15).json()
            except Exception as e:
                print(f"  [네트워크] {leg} {sid}: {e}")
                fail += len(ch)
                continue

            if resp.get("code") != 0:
                # 산복도로가 네비 도로망에 없어 실패하는 경우, 그 자체가 논거가 된다.
                # 버리지 말고 기록한다.
                print(f"  [code={resp.get('code')}] {leg} {sid}: {resp.get('message')}")
                for r in ch.itertuples():
                    f.write(json.dumps({"leg": leg, "start_id": sid, "goal_id": r.goal_id,
                                        "ok": False, "code": resp.get("code")},
                                       ensure_ascii=False) + "\n")
                fail += len(ch)
                time.sleep(SLEEP)
                continue

            for r, item in zip(ch.itertuples(), resp["route"][OPTION]):
                sm = item["summary"]
                f.write(json.dumps({"leg": leg, "start_id": sid, "goal_id": r.goal_id,
                                    "ok": True, "dist_m": sm["distance"],
                                    "dur_s": sm["duration"] / 1000.0,   # 응답은 밀리초
                                    "path": item["path"]}, ensure_ascii=False) + "\n")
                ok += 1
            f.flush()
            time.sleep(SLEEP)
    f.close()
    print(f"수집 완료: 성공 {ok} / 실패 {fail} → {RAW}")


# ============================================================================
# 3. 경로 표고 → 회귀
# ============================================================================
def path_profile(path, ef):
    """네비 경로 [[lon,lat],...] → (누적오르막 m, 누적내리막 m)"""
    a = np.asarray(path, dtype=float)
    x, y = TO_M.transform(a[:, 0], a[:, 1])
    dz = np.diff(ef(np.asarray(x), np.asarray(y)))
    return float(dz[dz > 0].sum()), float(-dz[dz < 0].sum())


def regress():
    import statsmodels.formula.api as smf

    ef = build_elev_interp(load_graph())
    rows = []
    for line in open(RAW, encoding="utf-8"):
        r = json.loads(line)
        if not r.get("ok"):
            rows.append(dict(leg=r["leg"], ok=False))
            continue
        up, down = path_profile(r["path"], ef)
        rows.append(dict(leg=r["leg"], start_id=r["start_id"], goal_id=r["goal_id"],
                         ok=True, dist_m=r["dist_m"], dur_s=r["dur_s"],
                         up_m=up, down_m=down))
    d = pd.DataFrame(rows)
    d.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"{OUT} 저장: 성공 {int(d.ok.sum())} / 실패 {int((~d.ok).sum())}")

    ok = d[d.ok]
    if len(ok) < 30:
        print("표본이 부족해 회귀를 건너뜁니다.")
        return None

    # ── (1) 절대 오르막 회귀 — 실패 사례로 남겨둔다 ────────────────────────
    # dur_s ~ dist_m + up_m 은 계수가 음수로 나온다(오르막이 빨라진다는 뜻).
    # 원인: 긴 경로일수록 간선도로라 빠른데 절대 오르막(m)도 함께 커져,
    #       up_m 이 '간선도로 여부'의 대리변수가 되어 부호가 뒤집힌다.
    m = smf.ols("dur_s ~ dist_m + up_m + down_m", data=ok).fit()
    print("\n" + "=" * 72)
    print("[참고] 절대 오르막 회귀 dur_s ~ dist_m + up_m + down_m")
    print(f"  up_m 계수 {m.params['up_m']:+.3f} (거리와 교란되어 부호 반전) "
          f"· R²={m.rsquared:.3f}")

    # ── (2) 평균경사 회귀 — 채택 모형 ──────────────────────────────────────
    # 거리로 나눠 '평균 경사'로 바꾸고, log(거리)를 도로등급 대리변수로 통제한다.
    #   소요초/거리 = c0 + b_up·(오르막/거리) + b_down·(내리막/거리) + c1·log(거리)
    # 양변에 거리를 곱하면  소요초 = c0·d + b_up·오르막 + ...  이므로
    # (오르막/거리)의 계수가 곧 b_up [s/m] 이다.
    # 대상: 대상지 내 A_out·A_back (이송 B는 간선 장거리라 체계가 다름 → 별도 확인)
    ab = ok[ok.leg.isin(["A_out", "A_back"])].copy()
    ab["gu"] = ab.up_m / ab.dist_m
    ab["gd"] = ab.down_m / ab.dist_m
    ab["tpm"] = ab.dur_s / ab.dist_m
    m2 = smf.ols("tpm ~ gu + gd + np.log(dist_m)", data=ab).fit()
    print("\n" + "=" * 72)
    print("[채택] 평균경사 회귀 tpm ~ gu + gd + log(dist)")
    print(m2.summary().tables[1])
    print(f"  R² = {m2.rsquared:.3f}   n = {int(m2.nobs)}")
    bu, bd = m2.params["gu"], m2.params["gd"]
    ci = m2.conf_int().loc["gu"]
    print(f"\n▶ b_up   = {bu:.3f} s/m  (p={m2.pvalues['gu']:.2g}, "
          f"95%CI [{ci[0]:.3f}, {ci[1]:.3f}])  → 100m 오르면 {bu*100:.0f}초 추가")
    print(f"▶ b_down = {bd:.3f} s/m  (p={m2.pvalues['gd']:.2g})"
          f"{'  ← 유의하지 않음 → 0 으로 처리' if m2.pvalues['gd'] > 0.05 else ''}")
    print("\n  → 출동시간_추정.py 의 B_UP / B_DOWN, 민감도 범위(95%CI)에 반영")

    # ── (2) 왕복 짝 분석 ──────────────────────────────────────────────────
    # 같은 구간을 뒤집으면 오르막↔내리막이 서로 바뀐다.
    #   t_out  = a·d + b_up·U + b_down·D
    #   t_back = a·d + b_up·D + b_down·U
    #   Δt = t_out − t_back = (b_up − b_down)(U − D) = (b_up − b_down)·Δh
    # 거리·도로·교통량이 상쇄되므로 경사 효과만 남는 가장 깨끗한 추정이다.
    out = ok[ok.leg == "A_out"][["start_id", "goal_id", "dur_s", "dist_m", "up_m", "down_m"]]
    bck = ok[ok.leg == "A_back"][["start_id", "goal_id", "dur_s", "dist_m", "up_m", "down_m"]]
    bck = bck.rename(columns={"start_id": "goal_id", "goal_id": "start_id"})   # 키 정렬
    p = out.merge(bck, on=["start_id", "goal_id"], suffixes=("_o", "_b"))
    print("\n" + "=" * 72)
    print(f"[왕복 짝 분석] 짝지어진 구간 {len(p)}쌍")

    b_up = b_down = None
    if len(p) >= 30:
        p = p.assign(dt=p.dur_s_o - p.dur_s_b,
                     dh=(p.up_m_o - p.down_m_o),          # 순 표고차(오르막 방향 기준)
                     dd=p.dist_m_o - p.dist_m_b)
        mp = smf.ols("dt ~ dh + dd", data=p).fit()
        print(mp.summary().tables[1])
        diff = mp.params.get("dh")
        print(f"\n  b_up − b_down = {diff:.3f} s/m   (p={mp.pvalues.get('dh'):.4f}, "
              f"R²={mp.rsquared:.3f})")
        print(f"  경로 길이 차이 중앙값 {p.dd.abs().median():.0f}m "
              f"(작을수록 같은 길로 왕복했다는 뜻)")

        if mp.pvalues.get("dh", 1) < 0.05 and diff > 0:
            # 내리막은 오르막의 약 1/4 로 두고 분해한다(감속하되 훨씬 작음).
            b_up, b_down = diff / 0.75, diff / 0.75 * 0.25
            print(f"\n▶ b_up   = {b_up:.3f} s/m   (1m 오를 때 추가 소요 초)")
            print(f"▶ b_down = {b_down:.3f} s/m")
            print(f"\n  → 출동시간_추정.py 의 B_UP_PROVISIONAL 을 {b_up:.2f} 로 교체")
        else:
            print("\n  ⚠ 짝 분석에서는 경사 효과가 분리되지 않는다.")
            print("     설계는 '같은 길 왕복'을 전제했으나 실제 경로 길이 차이가 크다(위 값 참조).")
            print("     일방통행이 많아 복귀 경로가 크게 우회하기 때문이다.")
            print("     → 경사계수는 위 [채택] 평균경사 회귀 결과를 사용한다.")
    else:
        print("  짝이 부족하다. A_out·A_back 을 모두 수집했는지 확인할 것.")
    return m, (b_up, b_down)


def test_one():
    """본 수집(212회) 전에 키·좌표순서·응답형식을 1건으로 확인한다.

    네이버는 '경도,위도' 순서를 쓴다. 뒤집으면 엉뚱한 좌표로 조회되므로
    거리·소요시간이 상식적인지 반드시 눈으로 확인할 것.
    """
    import requests

    kid, key = read_key("NCP_KEY_ID"), read_key("NCP_KEY")
    if not kid or not key:
        raise SystemExit("NCP_KEY_ID / NCP_KEY 를 .env 에 넣어주세요.")

    if not Path(OD).exists():
        build_od()
    od = pd.read_csv(OD, encoding="utf-8-sig")
    r = od[od.leg == "A_out"].iloc[0]

    print(f"테스트: {r.start_id} → 집계구 {r.goal_id}")
    print(f"  start = {r.start_lon},{r.start_lat}  (경도,위도 순서)")
    print(f"  goal  = {r.goal_lon},{r.goal_lat}")

    H = {"x-ncp-apigw-api-key-id": kid, "x-ncp-apigw-api-key": key}
    P = {"start": f"{r.start_lon},{r.start_lat}",
         "goal": f"{r.goal_lon},{r.goal_lat}", "option": OPTION}

    # 계정이 신규 Maps 인지 구버전 AI·NAVER API 인지에 따라 도메인이 다르다. 둘 다 시도한다.
    resp = requests.get(URL, headers=H, params=P, timeout=15)
    print(f"  {URL.split('/')[2]} → HTTP {resp.status_code}")
    if resp.status_code != 200:
        alt = requests.get(URL_LEGACY, headers=H, params=P, timeout=15)
        print(f"  {URL_LEGACY.split('/')[2]} → HTTP {alt.status_code}")
        if alt.status_code == 200:
            print("  ※ 구버전 도메인이 동작합니다. 스크립트 상단 URL 을 URL_LEGACY 값으로 바꾸세요.")
            resp = alt

    if resp.status_code != 200:
        # API Gateway 인증 실패는 route/code 가 아니라 error 객체로 온다. 본문을 그대로 보여준다.
        print(f"  ✗ 응답 본문: {resp.text[:400]}")
        print(f"  ✗ 사용한 KEY_ID 길이 {len(kid)}자, SECRET 길이 {len(key)}자")
        print("\n  [errorCode 해석]")
        print("    200 Authentication Failed → 키 값이 틀림(ID/Secret 뒤바뀜 포함)")
        print("    210 Permission Denied     → Application 에 Directions 5 미선택")
        print("    300 Not Found Service     → 상품 미신청")
        print("  * 방금 등록했다면 반영에 몇 분 걸릴 수 있습니다.")
        return False

    j = resp.json()
    if j.get("code") != 0:
        print(f"  ✗ code={j.get('code')}  {j.get('message')}")
        print("    좌표 이상 / 경로 없음")
        return False

    s = j["route"][OPTION][0]["summary"]
    dist, dur = s["distance"], s["duration"] / 1000
    print(f"  ✓ 거리 {dist:,}m · 소요 {dur/60:.1f}분 · 평균 {dist/dur*3.6:.1f} km/h")
    print(f"    경로점 {len(j['route'][OPTION][0]['path'])}개")
    if dist > 20000:
        print("  ⚠ 거리가 20km를 넘습니다. 좌표 순서가 뒤집혔을 가능성이 큽니다.")
        return False
    print("\n  정상입니다. 이제 전체 수집을 실행하세요:")
    print("    python code/scripts/검증_네비게이션.py")
    return True


if __name__ == "__main__":
    if "--test" in sys.argv:
        test_one()
    elif "--od-only" in sys.argv:
        build_od()
    elif "--regress-only" in sys.argv:
        regress()
    else:
        build_od()
        collect()
        regress()
