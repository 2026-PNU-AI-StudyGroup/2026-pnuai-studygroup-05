# -*- coding: utf-8 -*-
"""출동시간 추정 — 도달지연을 유효거리(m)에서 시간(분)으로 확장 + 3구간 분리

`경사비용_AED_추정.py`는 오르막 페널티를 준 **유효거리**(cost = L·(1+β·g))를 산출한다.
docs/02 말미의 "향후 도로등급별 속도모델로 시간(min) 산출 가능"을 구현한 것이 본 스크립트다.

  지연 = (경사 반영 시간) − (경사를 0으로 둔 시간)

이 차이가 "평지 기준으로 설계된 119 배치가 산복도로에서 보는 손해"이며,
개발배경 2번(평지 기준 응급체계의 부산 적용 한계)의 정량화다.

3구간 분리 (8/10 회의 「소방 → 발생 → 병원」)
  t1 출동  안전센터 → 하차지점      차량
  t2 접근  하차지점 → 집계구 중심   도보·들것   ← 기존 파이프라인에 없던 구간
  t3 이송  하차지점 → 병원          차량

기존 파이프라인 대비 보정 3가지
  1) 경사 클립 ±25%
     SRTM 30m 표고가 정수 미터로 반환돼, 짧은 엣지에서 1m 반올림 오차가 그대로 큰 경사가 된다.
     실측 예: 길이 2.1m·표고차 1m → 48%,  길이 11.6m·표고차 7m → 60%.
     실존 최급경사 도로가 약 35%이므로 물리적으로 불가능한 값이다.
  2) 하차지점을 최근접 '엣지'로 계산
     노트북 10은 `ox.distance.nearest_nodes`로 교차로 노드에 스냅한다. 구급차는 도로 선분
     어디에나 정차할 수 있으므로 점–선분 거리가 맞다. 노드 기준은 도보거리를 과대평가한다.
  3) 폭 필터 부재 우회
     `amb_passable`이 12,886개 엣지 전부 True이고 `width_est` 최소가 4.0m라 3m 미만이 0개다.
     도로등급별 일괄 부여값이라 실측 폭이 아니며, 노트북 10 라우팅에도 쓰이지 않는다
     (weight="length"만 사용). 폭으로 골목을 거를 수 없으므로, 하차지점–집계구 중심의
     실제 이격을 들것 도보구간으로 계상해 우회한다.

실행:  python code/scripts/출동시간_추정.py
출력:  outputs/oa_delay.csv
"""
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

# ── 저장소 루트로 이동: 실행 위치와 무관하게 data/·outputs/ 상대경로 유지 ──
for _c in [Path.cwd(), *Path.cwd().parents]:
    if (_c / "README.md").exists() and (_c / ".gitignore").exists():
        os.chdir(_c)
        break

import numpy as np
import pandas as pd
import networkx as nx
import osmnx as ox
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from shapely.geometry import Point, LineString
from shapely.strtree import STRtree

# ============================================================================
# 파라미터 (민감도 분석은 이 값만 바꾸면 된다)
# ============================================================================
GRADE_CLIP = 0.25          # 경사 클립 ±25%. 민감도 0.20 / 0.25 / 0.30

# 경사계수: 오르막 1m 당 추가 소요 초.  ★ 네비게이션 실측값 (검증_네비게이션.py)
#   모형:  소요초/거리m ~ (오르막/거리) + (내리막/거리) + log(거리)
#   결과:  오르막 0.489 s/m (p=6.5e-06)  ·  내리막 -0.043 s/m (p=0.67, 유의하지 않음)
#   n=1,091 (대상지 내 안전센터↔집계구 왕복 구간)
#   내리막은 유의하지 않아 0 으로 둔다. 100m 오르면 약 49초가 더 걸린다는 뜻.
B_UP = 0.489
B_DOWN = 0.0
# 하위호환용 별칭
B_UP_PROVISIONAL, B_DOWN_PROVISIONAL = B_UP, B_DOWN

# 도로등급별 기준속도 (km/h)
# `maxspeed` 태그가 12,886개 중 70개(0.5%)에만 있어 등급 기반으로 직접 정의한다.
V_BASE = {
    "motorway": 80, "trunk": 60, "primary": 50, "secondary": 40,
    "tertiary": 35, "residential": 25, "living_street": 15,
    "unclassified": 25, "busway": 30, "service": 20,
}
V_LINK_PENALTY = 10        # *_link(램프)는 본선보다 10km/h 낮게
V_DEFAULT = 25

# ★ 속도 보정계수 (네비게이션 실측 기반)
# V_BASE 는 도로등급별 '설계속도'라 실제 산복도로 주행속도와 크게 다르다.
# 안전센터→집계구 78개 구간에서 네비 실측과 대조한 결과, 보정 전 모델은 약 2.4배 빨랐다
#   보정 전 모델 1.50분(k 제외)  vs  네비 실측 3.67분
# 신호대기·교차로·좁은 도로·정지발진이 등급 기준속도에 반영되지 않기 때문이다.
# 이 계수를 곱하면 모델의 평지 주행시간이 일반 승용차(네비) 수준과 일치하고,
# 여기에 K_EMERGENCY 를 곱해 긴급주행을 반영한다.
V_CALIB = 0.34             # 실효속도 = V_BASE × 0.34  (예: residential 25 → 8.5km/h)

# 긴급주행 보정: 사이렌 주행은 신호대기·교차로 지연이 줄어든다.
# 축5(119 출동기록)에 시각 필드가 없어 실측 불가 → 문헌 범위 중앙값 + 민감도로 방어.
K_EMERGENCY = 0.70         # 민감도 0.60 / 0.70 / 0.80

# 도보(들것) 파라미터
WALK_LOAD_FACTOR = 0.60    # 들것·환자·장비 하중에 의한 보행속도 감소
WALK_SLOPE_CAP = 0.60      # 보행 경사 상한. 계단이라도 이보다 가파를 수 없다.
ONSCENE_SEC = 240.0        # 현장 처치 고정시간. 경사와 무관한 상수라 지연에서 상쇄된다.
TURNOUT_SEC = 90.0         # 신고접수 → 출동 준비(turnout)

# 구급차 제원 (특수구급차, 쏠라티급)
AMB = dict(m=4500, P=125_000, eta=0.85, C_rr=0.012, rho=1.225, CdA=1.8)
MU = {"마른 아스팔트": 0.80, "젖은 노면": 0.50, "결빙": 0.15}
DRIVE_AXLE_LOAD = 0.50     # 후륜구동, 구동륜 하중비

GRAPH = "outputs/graph_drive_conn.graphml"
OA = "outputs/oa_risk.parquet"
OUT = "outputs/oa_delay.csv"


def _sf(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


# ============================================================================
# 1. 견인력–저항 평형 (crawl speed) — 8/10 회의 요구 모델
# ============================================================================
def crawl_speed(grade, m, P, eta, C_rr, rho, CdA):
    """구동력 = 경사저항 + 구름저항 + 공기저항 이 되는 평형속도(m/s).

    0.5·rho·CdA·v³ + m·g·(sinθ + C_rr·cosθ)·v − P·eta = 0  의 양의 실근.
    """
    th = np.arctan(grade)
    R = m * 9.81 * (np.sin(th) + C_rr * np.cos(th))
    roots = np.roots([0.5 * rho * CdA, 0.0, R, -P * eta])
    pos = [r.real for r in roots if abs(r.imag) < 1e-6 and r.real > 0]
    return max(pos) if pos else 0.0


def physics_check():
    """출력이 제약인지 검증한다. 결론은 '아니다' — 원인을 배제하는 논거."""
    print("=" * 72)
    print("[검증] 견인력–저항 평형 모델")
    print("=" * 72)
    print(f"  구급차 {AMB['m']}kg / {AMB['P']/1000:.0f}kW = "
          f"{AMB['m']/(AMB['P']/1000):.0f} kg/kW   (AASHTO 화물차 기준 약 120 kg/kW)\n")
    print("  경사    평형속도    residential 기준 25km/h    제약 주체")
    for g in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        v = crawl_speed(g, **AMB) * 3.6
        print(f"  {g*100:4.0f}%  {v:7.1f} km/h    "
              f"{'출력' if v < 25 else '기하구조(도로폭·곡률)':<24}")
    print("\n  → 모든 경사에서 평형속도가 도로등급 기준속도를 상회한다.")
    print("     산복도로 지연의 원인은 동력 부족이 아니라 기하구조·견인한계·들것 도보구간이다.\n")
    print("  [견인한계] 등판 가능 최대 경사 = μ × 구동륜 하중비")
    for k, mu in MU.items():
        print(f"    {k:12s} {mu*DRIVE_AXLE_LOAD*100:5.1f}%")
    print("    → 결빙 시 7.5% 초과 구간 진입 불가. 겨울 산복도로 시나리오.\n")


# ============================================================================
# 2. 표고 보간 · 시간 가중치
# ============================================================================
def build_elev_interp(G):
    """그래프 노드 표고(4,756점, EPSG:5186)로 임의 좌표의 표고를 보간한다.

    별도 DEM 을 새로 받지 않는 이유: 노드가 곧 도로 교차점이라 도로 위 표고를 뽑기에 적합하고,
    docs/04 의 표고격자 레이어(③e)와 같은 방식이라 기존 결과와 정합한다.
    """
    pts, vals = [], []
    for _, d in G.nodes(data=True):
        if d.get("elev") == d.get("elev") and d.get("elev") is not None:
            pts.append((float(d["x"]), float(d["y"])))
            vals.append(float(d["elev"]))
    pts, vals = np.array(pts), np.array(vals)
    lin = LinearNDInterpolator(pts, vals)      # 볼록껍질 내부: 선형보간
    nea = NearestNDInterpolator(pts, vals)     # 껍질 바깥(병원 등): 최근접

    def f(x, y):
        x, y = np.atleast_1d(x), np.atleast_1d(y)
        z = lin(x, y)
        m = np.isnan(z)
        if m.any():
            z[m] = nea(x[m], y[m])
        return z
    return f


def v_base_of(hw):
    if hw is None:
        return V_DEFAULT
    hw = str(hw)
    if hw.endswith("_link"):
        return max(V_BASE.get(hw[:-5], V_DEFAULT) - V_LINK_PENALTY, 15)
    return V_BASE.get(hw, V_DEFAULT)


def add_time_weights(G, b_up, b_down, grade_clip=None, k=None):
    """엣지에 t_slope(경사반영)·t_flat(경사0) 초 단위 가중치를 부여.

    기본값을 인자 기본값으로 두면 정의 시점에 고정돼 민감도 분석에서 전역값을 바꿔도
    반영되지 않는다. None 으로 두고 호출 시점에 조회한다.
    """
    grade_clip = GRADE_CLIP if grade_clip is None else grade_clip
    k = K_EMERGENCY if k is None else k
    n_clip = 0
    for _, _, d in G.edges(data=True):
        L = float(d.get("length", 0) or 0)
        g = d.get("grade")
        g = g if (isinstance(g, float) and g == g) else 0.0
        if abs(g) > grade_clip:
            g = np.sign(g) * grade_clip
            n_clip += 1
        t_flat = L / (v_base_of(d.get("hw_type")) * V_CALIB / 3.6)
        dz = g * L                                   # 이 엣지의 표고 변화(m)
        t_slope = t_flat + (b_up * dz if dz > 0 else b_down * (-dz))
        d["t_flat"] = t_flat * k
        d["t_slope"] = t_slope * k
    return n_clip


# ============================================================================
# 3. t2 — 들것 도보 구간
# ============================================================================
def tobler_walk_speed(slope):
    """Tobler 보행함수(m/s). 차량용이 아니므로 t2 에만 쓴다."""
    return (6.0 * np.exp(-3.5 * np.abs(slope + 0.05))) / 3.6


def t2_stretcher(dx_m, dz_m):
    """하차지점 ↔ 집계구 중심 도보 왕복(초). 현장 고정시간은 포함하지 않는다."""
    dx = max(float(dx_m), 5.0)
    slope = float(np.clip(float(dz_m) / dx, -WALK_SLOPE_CAP, WALK_SLOPE_CAP))
    v_up = tobler_walk_speed(slope) * WALK_LOAD_FACTOR      # 올라갈 때(장비 지참)
    v_dn = tobler_walk_speed(-slope) * WALK_LOAD_FACTOR     # 내려올 때(환자 이송)
    dist = float(np.hypot(dx, dz_m))
    return dist / v_up + dist / v_dn


def nearest_edge_access(G, oa):
    """집계구 중심에서 가장 가까운 도로 '선분'까지의 하차거리와 하차지점 좌표."""
    nx_, ny_ = {}, {}
    for n, d in G.nodes(data=True):
        nx_[n], ny_[n] = float(d["x"]), float(d["y"])

    lines, meta = [], []
    for u, v, d in G.edges(data=True):
        geom = d.get("geometry")
        ls = geom if isinstance(geom, LineString) else \
            LineString([(nx_[u], ny_[u]), (nx_[v], ny_[v])])
        lines.append(ls)
        meta.append((u, v))

    tree = STRtree(lines)
    rows = []
    for cx, cy in zip(oa.cx.values, oa.cy.values):
        p = Point(cx, cy)
        i = tree.nearest(p)
        ls, (u, v) = lines[i], meta[i]
        q = ls.interpolate(ls.project(p))            # 선분 위 하차지점
        rows.append((p.distance(ls), q.x, q.y, u, v))
    return pd.DataFrame(rows, columns=["access_dx_m", "qx", "qy", "node_a", "node_b"],
                        index=oa.index)


# ============================================================================
# 4. 메인
# ============================================================================
def main(b_up=B_UP_PROVISIONAL, b_down=B_DOWN_PROVISIONAL, verbose=True):
    if verbose:
        physics_check()

    G = ox.load_graphml(GRAPH, edge_dtypes={"length": float, "grade": _sf},
                        node_dtypes={"elev": _sf})
    n_clip = add_time_weights(G, b_up, b_down)
    if verbose:
        print(f"[가중치] 경사 클립 적용 {n_clip} / {G.number_of_edges()} 엣지 "
              f"({n_clip/G.number_of_edges()*100:.1f}%)\n")

    stations = [n for n, d in G.nodes(data=True) if d.get("node_type") == "station"]
    hospitals = [n for n, d in G.nodes(data=True) if d.get("node_type") == "hospital"]

    oa = pd.read_parquet(OA).drop(columns=["geometry"])

    def multi_source_min(sources, weight, graph):
        best = {}
        for s in sources:
            for n, v in nx.single_source_dijkstra_path_length(
                    graph, s, weight=weight).items():
                if v < best.get(n, np.inf):
                    best[n] = v
        return best

    # t1: 안전센터 → 현장 (정방향)
    t1_slope = multi_source_min(stations, "t_slope", G)
    t1_flat = multi_source_min(stations, "t_flat", G)
    # t3: 현장 → 병원 (병원에서 역방향 탐색 = 현장→병원 방향의 경사를 보존)
    Grev = G.reverse(copy=False)
    t3_slope = multi_source_min(hospitals, "t_slope", Grev)
    t3_flat = multi_source_min(hospitals, "t_flat", Grev)

    # 하차지점: 최근접 '선분'
    oa = pd.concat([oa, nearest_edge_access(G, oa)], axis=1)

    def best_of(mapping, a, b):
        return np.minimum(pd.Series(a).map(mapping).values,
                          pd.Series(b).map(mapping).values)

    oa["t1_s"] = best_of(t1_slope, oa.node_a, oa.node_b) + TURNOUT_SEC
    oa["t1_flat_s"] = best_of(t1_flat, oa.node_a, oa.node_b) + TURNOUT_SEC
    oa["t3_s"] = best_of(t3_slope, oa.node_a, oa.node_b)
    oa["t3_flat_s"] = best_of(t3_flat, oa.node_a, oa.node_b)

    # t2: 하차지점 ↔ 중심. 두 표고를 같은 추정기로 뽑아야 한다.
    # 엣지 선형보간과 노드 보간기를 섞으면 두 점이 붙어 있어도 표고차가 남아 경사가 발산한다.
    ef = build_elev_interp(G)
    oa["elev_center"] = ef(oa.cx.values, oa.cy.values)
    oa["z_drop"] = ef(oa.qx.values, oa.qy.values)
    oa["access_dz_m"] = oa.elev_center - oa.z_drop
    oa["t2_s"] = [t2_stretcher(dx, dz) for dx, dz in zip(oa.access_dx_m, oa.access_dz_m)]
    oa["t2_flat_s"] = [t2_stretcher(dx, 0.0) for dx in oa.access_dx_m]

    # 합산·지연. 현장처치 고정시간은 상수라 지연에서 상쇄되고 총시간에만 더해진다.
    oa["total_s"] = oa.t1_s + oa.t2_s + ONSCENE_SEC + oa.t3_s
    oa["total_flat_s"] = oa.t1_flat_s + oa.t2_flat_s + ONSCENE_SEC + oa.t3_flat_s
    oa["delay_s"] = oa.total_s - oa.total_flat_s

    # 구간별 지연 분해: 지연이 어디서 나는지가 정책 함의를 가른다.
    #   t1 크면 안전센터 배치 문제 / t2 크면 AED 선배치가 답 / t3 크면 병원 접근성 문제
    for seg in ["t1", "t2", "t3"]:
        oa[f"delay_{seg}_s"] = oa[f"{seg}_s"] - oa[f"{seg}_flat_s"]

    for c in ["t1", "t2", "t3", "total", "total_flat", "delay",
              "delay_t1", "delay_t2", "delay_t3"]:
        oa[f"{c}_min"] = oa[f"{c}_s"] / 60.0

    # 위험도 재산출: f_reach 를 거리 → 시간으로 교체.
    # 노트북 10과 동일하게 95백분위 클립 후 min-max 정규화한다(원본은 보존).
    cap = oa.t1_s.clip(upper=oa.t1_s.quantile(0.95))
    oa["f_reach_time"] = (cap - cap.min()) / (cap.max() - cap.min())
    r = oa.f_age * oa.f_reach_time
    oa["risk_norm_time"] = (r - r.min()) / (r.max() - r.min())

    out = oa[["TOT_OA_CD", "dong", "entry_node", "dist_station_m",
              "access_dx_m", "access_dz_m",
              "t1_min", "t2_min", "t3_min", "total_min", "total_flat_min", "delay_min",
              "delay_t1_min", "delay_t2_min", "delay_t3_min",
              "p75_2026", "f_age", "f_reach", "f_reach_time",
              "risk_norm", "risk_norm_time"]].copy()
    out.to_csv(OUT, index=False, encoding="utf-8-sig")

    if verbose:
        report(out, b_up, b_down)
    return out


def report(d, b_up, b_down):
    print("=" * 72)
    print(f"[결과] b_up={b_up} s/m (네비 실측) · b_down={b_down} · k={K_EMERGENCY}")
    print("=" * 72)
    print("\n구간별 소요 (분)")
    print(d[["t1_min", "t2_min", "t3_min", "total_min", "delay_min"]]
          .describe().loc[["min", "50%", "max"]].round(2).to_string())

    print("\n동별 평균")
    g = d.groupby("dong").agg(
        t1=("t1_min", "mean"), t2=("t2_min", "mean"), t3=("t3_min", "mean"),
        총시간=("total_min", "mean"), 지연=("delay_min", "mean"),
        위험도_거리=("risk_norm", "mean"), 위험도_시간=("risk_norm_time", "mean"),
    ).round(2).sort_values("위험도_시간", ascending=False)
    print(g.to_string())

    a = d.groupby("dong").risk_norm.mean().rank(ascending=False)
    b = d.groupby("dong").risk_norm_time.mean().rank(ascending=False)
    print("\n[순위 변동] 거리기준 → 시간기준")
    print(pd.DataFrame({"거리": a, "시간": b, "Δ": a - b})
          .sort_values("시간").to_string())
    print(f"\n집계구 순위 상관(Spearman): "
          f"{d.risk_norm.corr(d.risk_norm_time, method='spearman'):.3f}")
    i = d.total_min.idxmax()
    print(f"최악 집계구: {d.loc[i,'dong']} {d.total_min.max():.1f}분 "
          f"(지연 {d.loc[i,'delay_min']:.1f}분)")

    print("\n[지연의 출처] 구간별 분해 (분, 평균)")
    dec = d[["delay_t1_min", "delay_t2_min", "delay_t3_min"]].mean()
    for k, v in dec.items():
        print(f"  {k:16s} {v:5.2f}분 ({v/dec.sum()*100:4.1f}%)")


def sensitivity():
    """축5에 시각 필드가 없어 실측 검증이 불가하므로, 결론이 파라미터에
       얼마나 의존하는지를 보여 방어한다."""
    global GRADE_CLIP
    print("\n" + "=" * 72)
    print("[민감도 분석]")
    print("=" * 72)
    base = GRADE_CLIP
    rows = []
    # b_up 범위는 네비 회귀의 95% 신뢰구간 [0.277, 0.701] 을 그대로 쓴다.
    # 임의 배수가 아니라 추정 불확실성 자체를 민감도로 옮긴 것이라 방어가 쉽다.
    for b_up in [0.277, 0.489, 0.701]:
        for clip in [0.20, 0.25, 0.30]:
            GRADE_CLIP = clip
            d = main(b_up=b_up, verbose=False)
            rows.append(dict(b_up=b_up, clip=clip,
                             지연중앙=round(d.delay_min.median(), 2),
                             지연최대=round(d.delay_min.max(), 2),
                             총시간중앙=round(d.total_min.median(), 2),
                             최고위험동=d.groupby("dong").risk_norm_time.mean().idxmax(),
                             spearman=round(d.risk_norm.corr(
                                 d.risk_norm_time, method="spearman"), 3)))
    GRADE_CLIP = base
    r = pd.DataFrame(rows)
    print(r.to_string(index=False))
    print(f"\n최고위험 동이 모든 조합에서 동일: {r.최고위험동.nunique() == 1} "
          f"({list(r.최고위험동.unique())})")
    return r


if __name__ == "__main__":
    main()
    sensitivity()
    print(f"\n저장: {OUT}")
