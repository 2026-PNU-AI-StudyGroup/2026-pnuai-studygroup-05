# -*- coding: utf-8 -*-
"""07. 성능평가 — 반사실 개선량 (기존 AED 106 vs 통합)

docs/07 재현 스크립트. "신규 배치로 얼마나 좋아지는가"를 정량화한다.

  커버 = 골든타임 도보 왕복 T_gold(4분) + 표고차 빗변보정(√)  ← 경사 페널티계수 안 씀
  상태 A(기존106) vs C(기존+신규), 주간/야간 분리
  P1 인구·P2 위험도(별도) · P3 형평성(도달 4/6/8분 초과) · P4 야간Δ · P5 비용구조

경로: 저장소 루트 기준 상대경로 (팀 스크립트와 동일 방식).
입력:
  data/output/집계구별_출동시간_지연.csv      (도달시간 t1+t2)
  data/output/AED_신규배치_접근성.csv         (신규 후보 36, Tier·좌표·표고)
  data/aed/aed_donggu.csv                      (기존 AED 106, 운영시간) ※ 없으면 안내 후 종료
  outputs/oa_risk.parquet, graph_drive_conn.graphml  (좌표·표고, 노트북 재현 산출)
출력:
  data/output/평가_집계구별.csv
"""
import sys, os
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")

# ── 저장소 루트로 이동 (팀 방식과 동일) ──
for _c in [Path.cwd(), *Path.cwd().parents]:
    if (_c / "README.md").exists() and (_c / ".gitignore").exists():
        os.chdir(_c); break

import numpy as np, pandas as pd, geopandas as gpd
from scipy.spatial import cKDTree
from pyproj import Transformer
import osmnx as ox

# ── 파라미터 ──
T_GOLD, V_WALK, TAU = 4.0, 1.3, 1.3
TO_M = Transformer.from_crs(4326, 5186, always_xy=True)

def _sf(x):
    try: return float(x)
    except: return float("nan")

# ── 입력 경로 (상대) ──
P_DELAY = "data/output/집계구별_출동시간_지연.csv"
P_NEW   = "data/output/AED_신규배치_접근성.csv"
P_RISK  = "outputs/oa_risk.parquet"
P_GRAPH = "outputs/graph_drive_conn.graphml"
P_AED   = "data/aed/aed_donggu.csv"     # 기존 AED (개인정보 컬럼 제외본)

for p in [P_DELAY, P_NEW, P_RISK, P_GRAPH]:
    if not Path(p).exists():
        raise SystemExit(f"입력 없음: {p}\n  노트북(01·10)·docs06 스크립트를 먼저 실행해 산출물을 만드세요.")

# ══════════════════════════════════════════════════════════
# 1. 집계구 (수요·좌표·표고·도달시간)
# ══════════════════════════════════════════════════════════
oa = gpd.read_parquet(P_RISK)[["TOT_OA_CD","dong","cx","cy","p75_2026","risk_norm","entry_node"]].copy()
oa["TOT_OA_CD"] = oa["TOT_OA_CD"].astype(str)
G = ox.load_graphml(P_GRAPH, node_dtypes={"elev": _sf})
oa["elev"] = [G.nodes[n].get("elev", np.nan) for n in oa["entry_node"]]
dt = pd.read_csv(P_DELAY, encoding="utf-8-sig")
dt["집계구코드"] = dt["집계구코드"].astype(str)
dt["구급차도달_분"] = dt["t1_출동_분"] + dt["t2_접근_들것_분"]
oa = oa.merge(dt[["집계구코드","구급차도달_분"]], left_on="TOT_OA_CD", right_on="집계구코드", how="left")
print(f"집계구 {len(oa)} · 75+ 총 {oa.p75_2026.sum():.0f}명")

# ══════════════════════════════════════════════════════════
# 2. AED — 기존 106(있으면) + 신규 36
# ══════════════════════════════════════════════════════════
if Path(P_AED).exists():
    aed = pd.read_csv(P_AED, encoding="utf-8-sig").dropna(subset=["wgs84Lat","wgs84Lon"])
    def is_night(r):
        try: e = int(float(r.get("monEndTme")))
        except: return False
        s = str(r.get("monSttTme","")).strip().replace(".0","")
        return e >= 2200 or (s in ("0","0000","") and e in (0,2400,2359))
    aed["night"] = aed.apply(is_night, axis=1)
    ax, ay = TO_M.transform(aed["wgs84Lon"].values, aed["wgs84Lat"].values)
    aed["x"], aed["y"] = ax, ay
    aed["elev"] = [G.nodes[n].get("elev", np.nan) for n in ox.distance.nearest_nodes(G, X=ax, Y=ay)]
    aed = aed[["x","y","elev","night"]]
    print(f"기존 AED {len(aed)} · 야간 {int(aed.night.sum())} / 주간 {int((~aed.night).sum())}")
else:
    aed = pd.DataFrame(columns=["x","y","elev","night"])
    print(f"[주의] {P_AED} 없음 → 기존 AED 미반영(baseline=0). 방침상 원본 미공개 시 로컬에서만 완전재현.")

nw = pd.read_csv(P_NEW, encoding="utf-8-sig")
nx_, ny_ = TO_M.transform(nw["lon"].values, nw["lat"].values)
nw["x"], nw["y"], nw["elev"], nw["night"] = nx_, ny_, nw["표고_m"], True
print(f"신규 후보 {len(nw)} · Tier {nw.유형.value_counts().to_dict()}")

# ══════════════════════════════════════════════════════════
# 3. 커버 판정 (빗변보정 도보 왕복 ≤ T_gold)
# ══════════════════════════════════════════════════════════
def covered_mask(df, night_only, t_gold=T_GOLD):
    sub = df[df.night] if night_only else df
    if len(sub) == 0: return np.zeros(len(oa), bool)
    pts = np.c_[sub.x.values, sub.y.values]; el = sub.elev.values
    tree = cKDTree(pts)
    r = V_WALK * t_gold * 60 / 2 / TAU * 1.2
    cov = np.zeros(len(oa), bool)
    for i, (cx, cy, ce) in enumerate(zip(oa.cx, oa.cy, oa.elev)):
        for j in tree.query_ball_point([cx, cy], r):
            dz = abs(ce - el[j]) if (ce == ce and el[j] == el[j]) else 0.0
            d_eff = np.hypot(np.hypot(cx-pts[j,0], cy-pts[j,1]), dz)
            if 2 * (d_eff * TAU) / V_WALK / 60 <= t_gold:
                cov[i] = True; break
    return cov

comb = pd.concat([aed, nw[["x","y","elev","night"]]], ignore_index=True)
scen = {}
for lab, df in [("A", aed), ("C", comb)]:
    for sc, no in [("주간", False), ("야간", True)]:
        c = covered_mask(df, no)
        scen[f"{lab}_{sc}"] = dict(n=int(c.sum()), p75=float(oa.p75_2026[c].sum()),
                                    risk=float(oa.risk_norm[c].sum()), cov=c)

# ── 출력 ──
tot = oa.p75_2026.sum()
print("\n[P1·P2] 커버 (A=기존, C=통합)")
for k, v in scen.items():
    print(f"  {k:8s} 집계구 {v['n']:2d}/78 · 75+ {v['p75']:5.0f}({v['p75']/tot*100:4.1f}%) · 위험도 {v['risk']:5.2f}")
print("\n[P4] 개선 Δ = C−A")
for sc in ["주간","야간"]:
    a, c = scen[f"A_{sc}"], scen[f"C_{sc}"]
    print(f"  {sc}: 75+ +{c['p75']-a['p75']:.0f}명(+{(c['p75']-a['p75'])/tot*100:.1f}%p) · 집계구 +{c['n']-a['n']}")

print("\n[P3] 형평성 — 도달 초과 취약지 야간커버율")
cA, cC = scen["A_야간"]["cov"], scen["C_야간"]["cov"]
for thr in [4,6,8]:
    v = oa["구급차도달_분"] > thr; nv = int(v.sum())
    if nv == 0: continue
    print(f"  >{thr}분({nv}개,75+{oa.p75_2026[v].sum():.0f}): {((cA&v).sum()/nv*100):.1f}% → "
          f"{((cC&v).sum()/nv*100):.1f}% (Δ{((cC&v).sum()-(cA&v).sum())/nv*100:+.1f}%p)")

print("\n[P5] 비용구조 — 무비용(Tier1)부터")
cum = aed.copy(); prev = oa.p75_2026[scen["A_야간"]["cov"]].sum(); base = prev
print(f"  기존만: 75+ {prev:.0f}명")
for t in ["Tier1","Tier2","Tier3"]:
    cum = pd.concat([cum, nw[nw.유형==t][["x","y","elev","night"]]], ignore_index=True)
    now = oa.p75_2026[covered_mask(cum, True)].sum()
    print(f"  +{t}({int((nw.유형==t).sum())}개): 75+ {now:.0f} · 순증 +{now-prev:.0f} · 누적Δ +{now-base:.0f}")
    prev = now

print("\n[보수 시나리오] 실존시설(Tier1·2)만 vs 전체 — 야간")
t12 = nw[nw.유형.isin(["Tier1","Tier2"])][["x","y","elev","night"]]
for lab, df in [("기존106", aed),
                ("기존+실존(T1·2,12곳)", pd.concat([aed, t12], ignore_index=True)),
                ("기존+전체(36곳)", comb)]:
    c = covered_mask(df, True); p = oa.p75_2026[c].sum()
    print(f"  {lab:22s}: 75+ {p:5.0f}({p/tot*100:4.1f}%)")
c12 = covered_mask(pd.concat([aed, t12], ignore_index=True), True)
print("  형평성(실존만): ", end="")
for thr in [4,6,8]:
    v = oa["구급차도달_분"] > thr; nv = int(v.sum())
    print(f">{thr}분 Δ{((c12&v).sum()-(cA&v).sum())/nv*100:+.0f}%p ", end="")
print("← 8분초과=0%p면 신설 불가피")

print("\n[역검증] 표고 중앙")
if len(aed) and aed.night.any():
    print(f"  기존 야간AED {aed[aed.night].elev.median():.0f}m · 신규 {nw.elev.median():.0f}m · 고위험집계구 {oa.nlargest(20,'risk_norm').elev.median():.0f}m")

# ── 저장 ──
out = oa[["TOT_OA_CD","dong","p75_2026","risk_norm","구급차도달_분","elev"]].copy()
out["커버_기존야간"] = scen["A_야간"]["cov"]
out["커버_통합야간"] = scen["C_야간"]["cov"]
out["신규커버_획득"] = (~scen["A_야간"]["cov"]) & scen["C_야간"]["cov"]
out.to_csv("data/output/평가_집계구별.csv", index=False, encoding="utf-8-sig")
print("\n저장: data/output/평가_집계구별.csv")
