# -*- coding: utf-8 -*-
"""반사실 검증 — "이 15개를 놓으면 무엇이 얼마나 좋아지는가".

자문의견서(2026-08-16)가 가장 강하게 지적한 대목이다.
  "현재 산출물은 '여기가 위험하다'에서 끝나는데, 정책 제안이라면
   '이 15개를 놓으면 무엇이 얼마나 좋아지는가'를 숫자로 제시할 수 있어야 한다."

무엇을 비교하는가.
  현재(AED 없음)  심정지 → 구급차를 기다림 → 제세동까지 t1+t2 분
  설치 후          목격자가 가장 가까운 AED 를 왕복해 옴 → 제세동
                   (구급차가 더 빠르면 구급차 시간을 그대로 쓴다)

핵심 지표는 '분'이다. 생존율 환산은 문헌 계수를 곱한 파생값이라
가정에 민감하므로 범위로만 병기한다.

실행:  python code/scripts/반사실_검증.py
출력:  data/output/반사실_집계구별.csv
"""
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

for _c in [Path.cwd(), *Path.cwd().parents]:
    if (_c / "README.md").exists() and (_c / ".gitignore").exists():
        os.chdir(_c)
        break

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree

# ============================================================================
# 파라미터 — 전부 근거를 붙인다
# ============================================================================
GOLDEN = 4.0        # 골든타임(분). 4~6분 중 보수적으로 짧은 쪽.
WALK_MS = 1.4       # 목격자 보행속도(m/s). 성인 빠른 걸음.
SLOPE_K = 1.20      # 경사 유효거리 배수. 본 프로젝트 실측(오르막 페널티 평균 ×1.20).
PREP_MIN = 1.0      # AED 꺼내 패드 부착까지. 보수적으로 1분.
R_REACH = 400.0     # 목격자가 가지러 갈 만한 최대 거리(m). 왕복 유효 960m ≈ 11분.

# 제세동 1분 지연당 생존율 감소폭(%p). 대한심폐소생협회·AHA 가이드라인 통용 범위.
SURV_LOSS_LO, SURV_LOSS_HI = 7.0, 10.0

OUT = "data/output/반사실_집계구별.csv"


def main():
    dem = pd.read_csv("data/output/집계구별_출동시간_지연.csv", dtype={"집계구코드": str})
    dem["구급차_분"] = dem["t1_출동_분"] + dem["t2_접근_들것_분"]

    # 집계구 좌표(면적 플래그 포함)는 배치 스크립트와 같은 방식으로 얻는다
    sys.path.insert(0, "code/scripts")
    from importlib import import_module
    place = import_module("AED_실좌표_배치")
    oa = place.load_tracts()
    dem = dem.merge(oa, on="집계구코드", how="left")
    med = dem["면적_ha"].median()
    dem["산지포함_의심"] = dem["면적_ha"] > med * 5

    new = pd.read_csv("data/output/AED_실좌표_후보.csv")

    to_m = Transformer.from_crs("EPSG:4326", "EPSG:5186", always_xy=True)
    dx, dy = to_m.transform(dem.oa_lon.values, dem.oa_lat.values)
    ax, ay = to_m.transform(new.lon.values, new.lat.values)

    d, idx = cKDTree(np.c_[ax, ay]).query(np.c_[dx, dy])
    dem["최근접AED_m"] = d
    dem["최근접AED"] = new.이름.to_numpy()[idx]

    # ── 설치 후 제세동까지 시간 ────────────────────────────────────────────
    # 왕복 유효거리 = 2 × 직선거리 × 경사배수
    walk_min = (2 * d * SLOPE_K) / WALK_MS / 60.0
    aed_min = walk_min + PREP_MIN
    # 너무 멀면 아무도 가지러 가지 않는다 → 구급차를 기다린다
    aed_min = np.where(d <= R_REACH, aed_min, np.inf)

    dem["AED_왕복_분"] = np.where(np.isfinite(aed_min), np.round(aed_min, 2), np.nan)
    dem["설치후_분"] = np.minimum(dem["구급차_분"], aed_min)
    dem["단축_분"] = (dem["구급차_분"] - dem["설치후_분"]).clip(lower=0)

    # ── 지표 ───────────────────────────────────────────────────────────────
    pop = dem["75세이상인구_2026"]
    tot = pop.sum()

    def within(col):
        return float(pop[dem[col] <= GOLDEN].sum())

    before_in = within("구급차_분")
    after_in = within("설치후_분")

    w_before = float((dem["구급차_분"] * pop).sum() / tot)
    w_after = float((dem["설치후_분"] * pop).sum() / tot)
    w_gain = w_before - w_after

    # 생존율은 파생 추정이라 범위로만
    surv_lo = w_gain * SURV_LOSS_LO
    surv_hi = w_gain * SURV_LOSS_HI

    dem.to_csv(OUT, index=False, encoding="utf-8-sig")

    print("=" * 74)
    print("반사실 검증 — 신규 AED %d개소 설치 전후" % len(new))
    print("=" * 74)
    print(f"대상: 75세 이상 {tot:,.0f}명 · 집계구 {len(dem)}개")
    print()
    print("① 제세동까지 걸리는 시간 (75세+ 인구 가중평균)")
    print(f"   설치 전  {w_before:6.2f}분   (구급차를 기다림)")
    print(f"   설치 후  {w_after:6.2f}분   (가까우면 AED 를 가지러 감)")
    print(f"   단축     {w_gain:6.2f}분   ({w_gain/w_before*100:.0f}% 감소)")
    print()
    print("② 골든타임 %.0f분 안에 제세동 가능한 75세+ 인구" % GOLDEN)
    print(f"   설치 전  {before_in:7,.0f}명  ({before_in/tot*100:5.1f}%)")
    print(f"   설치 후  {after_in:7,.0f}명  ({after_in/tot*100:5.1f}%)")
    print(f"   증가     {after_in-before_in:7,.0f}명  "
          f"(+{(after_in-before_in)/tot*100:.1f}%p)")
    print()
    print("③ 생존율 추정 (제세동 1분 지연당 %.0f~%.0f%%p 감소 가정)"
          % (SURV_LOSS_LO, SURV_LOSS_HI))
    print(f"   기대 개선  +{surv_lo:.1f} ~ +{surv_hi:.1f}%p")
    print("   ※ 문헌 계수를 곱한 파생 추정이므로 범위로만 제시한다.")
    print()
    n_help = int((dem["단축_분"] > 0).sum())
    print(f"④ 실제로 단축이 생긴 집계구 {n_help}/{len(dem)}개 "
          f"(최근접 AED {R_REACH:.0f}m 이내)")
    top = dem.nlargest(5, "단축_분")[
        ["행정동", "75세이상인구_2026", "구급차_분", "AED_왕복_분", "단축_분",
         "최근접AED", "산지포함_의심"]]
    print(top.round(2).to_string(index=False))
    mnt = dem[dem.산지포함_의심 & (dem.단축_분 > 0)]
    if len(mnt):
        print(f"\n   ⚠ 위 중 산지포함 의심 집계구 {len(mnt)}개는 중심점이 무인 산지라")
        print("     구급차 시간이 과대추정돼 단축폭도 부풀려져 있다(해석 주의).")
    print(f"\n저장: {OUT}")


if __name__ == "__main__":
    main()
