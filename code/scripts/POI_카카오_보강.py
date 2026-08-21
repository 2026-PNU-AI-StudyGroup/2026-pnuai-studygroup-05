# -*- coding: utf-8 -*-
"""카카오 로컬 API 로 AED 후보 시설(POI)을 보강한다.

왜 필요한가.
  OSM 은 한국 소규모·생활 시설의 등록률이 낮다. 실측으로 대상지(초량·좌천동)에서
  경로당이 단 1개만 잡혔는데, 산복도로에는 실제로 훨씬 많다. 그 결과
  "반경 150m 안에 후보 시설이 하나도 없다"고 판정된 집계구 중 상당수가
  실제로는 시설이 있는데 데이터에 없어서 공백으로 보였을 가능성이 크다.

무엇을 하는가.
  대상지를 격자로 쪼개(카카오는 한 질의당 최대 45건) 카테고리·키워드 검색을
  반복하고, 좌표 기준으로 중복을 제거해 POI 목록을 만든다.
  OSM 결과와 합쳐 AED_실좌표_배치.py 가 쓰는 후보 풀을 넓힌다.

실행:  python code/scripts/POI_카카오_보강.py
출력:  outputs/poi_kakao.json          (원본 응답 정리본)
       data/output/AED_후보시설_통합.csv (OSM + 카카오 병합·중복제거)
"""
import os
import sys
import json
import time
import math
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

for _c in [Path.cwd(), *Path.cwd().parents]:
    if (_c / "README.md").exists() and (_c / ".gitignore").exists():
        os.chdir(_c)
        break

import numpy as np
import pandas as pd
import requests

# ============================================================================
# 파라미터
# ============================================================================
BBOX = (35.1100, 129.0260, 35.1400, 129.0580)     # S, W, N, E
GRID = 5              # 한 변 분할 수. 5x5=25칸 → 칸당 반경 약 350m
RADIUS_PAD = 1.35     # 칸 대각선의 절반에 곱할 여유(칸 사이 빈틈 방지)
PAGE_MAX = 3          # 카카오는 페이지당 15건 · 최대 3페이지(45건)
SLEEP = 0.12          # 호출 간격(레이트리밋 여유)

OUT_JSON = "outputs/poi_kakao.json"
OUT_CSV = "data/output/AED_후보시설_통합.csv"

# 카테고리 그룹 코드 — 코드 검색이 키워드보다 누락이 적다.
CATEGORIES = {
    "CS2": "편의점",
    "MT1": "대형마트",
    "PO3": "공공기관",
    "OL7": "주유소",
    "SW8": "지하철역",
}
# 카테고리로 안 잡히는 것들은 키워드로 보완한다.
# 경로당·노인정은 고령 인구가 모이는 곳이라 이 프로젝트에서 특히 중요하다.
KEYWORDS = ["경로당", "노인정", "노인복지관", "주민센터", "행정복지센터",
            "파출소", "지구대", "치안센터", "우체국", "슈퍼", "마트"]

BASE = "https://dapi.kakao.com/v2/local/search"


def read_key():
    """.env 우선, 없으면 환경변수. 다른 스크립트와 같은 방식."""
    name = "KAKAO_REST_API_KEY"
    if Path(".env").exists():
        for line in open(".env", encoding="utf-8"):
            line = line.strip()
            if line.startswith(name + "="):
                v = line.split("=", 1)[1].strip().strip('"').strip("'")
                if v:
                    return v
    return os.environ.get(name)


def cells():
    """대상지를 GRID x GRID 로 나눠 (중심위경도, 반경m) 목록을 만든다."""
    S, W, N, E = BBOX
    dlat = (N - S) / GRID
    dlon = (E - W) / GRID
    # 위도 35도 부근: 위도 1도≈111km, 경도 1도≈91km
    half_m = 0.5 * math.hypot(dlat * 111_000, dlon * 91_000)
    r = int(min(20_000, half_m * RADIUS_PAD))
    out = []
    for i in range(GRID):
        for j in range(GRID):
            out.append((W + (j + 0.5) * dlon, S + (i + 0.5) * dlat, r))
    return out


def call(kind, key, params):
    url = f"{BASE}/{kind}.json"
    h = {"Authorization": f"KakaoAK {key}"}
    for attempt in range(3):
        try:
            r = requests.get(url, headers=h, params=params, timeout=20)
            if r.status_code == 401:
                raise SystemExit(
                    "카카오 인증 실패(401). REST API 키가 맞는지 확인하세요.\n"
                    "  developers.kakao.com → 내 애플리케이션 → 앱 키 → REST API 키\n"
                    "  (JavaScript 키·Admin 키가 아닙니다)")
            if r.status_code == 429:
                time.sleep(1.5)
                continue
            r.raise_for_status()
            return r.json()
        except SystemExit:
            raise
        except Exception:                                    # noqa: BLE001
            if attempt == 2:
                return None
            time.sleep(1.0)
    return None


def collect(key):
    rows, seen = [], set()

    def add(docs, src):
        for d in docs:
            try:
                lon, lat = float(d["x"]), float(d["y"])
            except (KeyError, TypeError, ValueError):
                continue
            # 좌표 6자리(약 0.1m)로 중복 판정
            k = (d.get("place_name", ""), round(lon, 6), round(lat, 6))
            if k in seen:
                continue
            seen.add(k)
            rows.append(dict(이름=d.get("place_name", ""),
                             카테고리=d.get("category_name", ""),
                             그룹=d.get("category_group_name", ""),
                             주소=d.get("road_address_name") or d.get("address_name", ""),
                             전화=d.get("phone", ""),
                             lon=lon, lat=lat, 수집=src))

    cl = cells()
    print(f"[카카오] 격자 {len(cl)}칸 × (카테고리 {len(CATEGORIES)} + 키워드 {len(KEYWORDS)})")
    for n, (x, y, r) in enumerate(cl, 1):
        for code, label in CATEGORIES.items():
            for page in range(1, PAGE_MAX + 1):
                js = call("category", key, dict(category_group_code=code, x=x, y=y,
                                                radius=r, page=page, size=15))
                if not js:
                    break
                add(js.get("documents", []), f"category:{label}")
                if js.get("meta", {}).get("is_end", True):
                    break
                time.sleep(SLEEP)
        for kw in KEYWORDS:
            for page in range(1, PAGE_MAX + 1):
                js = call("keyword", key, dict(query=kw, x=x, y=y, radius=r,
                                               page=page, size=15))
                if not js:
                    break
                add(js.get("documents", []), f"keyword:{kw}")
                if js.get("meta", {}).get("is_end", True):
                    break
                time.sleep(SLEEP)
        if n % 5 == 0 or n == len(cl):
            print(f"  {n}/{len(cl)}칸 · 누적 {len(rows)}건")
    return pd.DataFrame(rows)


# ============================================================================
# 분류 — AED_실좌표_배치.py 와 같은 기준으로 유형·야간접근을 매긴다
# ============================================================================
def classify_kakao(name, cat, group):
    t = f"{name} {cat} {group}"
    if any(k in t for k in ("병원", "의원", "한의", "치과", "요양", "약국")):
        return None                              # 의료기관 제외(자체 장비 보유)
    if "편의점" in t:
        brand = any(b in name.upper() for b in
                    ("CU", "GS25", "세븐일레븐", "이마트24", "미니스톱", "7-ELEVEN"))
        return ("편의점", 1.0 if brand else 0.4,
                1.0 if brand else 0.9,
                "없음(이미 24시간)" if brand else "운영시간 확인 후 야간 개방 협의")
    if any(k in t for k in ("경로당", "노인정", "노인복지")):
        return ("경로당", 0.1, 1.0, "★ 외벽 24시간 함체 설치")
    if any(k in t for k in ("주민센터", "행정복지센터")):
        return ("주민센터", 0.1, 1.0, "★ 외벽 24시간 함체 설치")
    if any(k in t for k in ("파출소", "지구대", "치안센터", "경찰")):
        return ("파출소·지구대", 1.0, 1.0, "없음(24시간 상주)")
    if any(k in t for k in ("소방서", "안전센터", "119")):
        return ("소방서·안전센터", 1.0, 1.0, "없음(24시간 상주)")
    if any(k in t for k in ("대형마트", "마트", "슈퍼", "시장")):
        return ("마트", 0.4, 0.6, "영업시간 외 외벽 함체 설치")
    if "주유소" in t:
        return ("주유소", 0.4, 0.8, "야간 무인 구역에 함체 설치")
    if "지하철" in t or "역" == name[-1:]:
        return ("도시철도역", 0.4, 0.7, "역사 외부 개방구역에 설치")
    if any(k in t for k in ("우체국", "공공기관", "구청", "도서관", "복지관")):
        return ("공공시설", 0.1, 1.0, "★ 외벽 24시간 함체 설치")
    return None


def main():
    key = read_key()
    if not key:
        raise SystemExit(
            "KAKAO_REST_API_KEY 가 없습니다.\n"
            "  1) developers.kakao.com → 내 애플리케이션 → 앱 생성\n"
            "  2) 앱 키 → 'REST API 키' 복사\n"
            "  3) 저장소 루트 .env 에  KAKAO_REST_API_KEY=키값  (따옴표 없이)\n")

    df = collect(key)
    S, W, N, E = BBOX
    df = df[(df.lat.between(S, N)) & (df.lon.between(W, E))].copy()
    print(f"\n[카카오] 대상지 내 고유 POI {len(df)}건")

    cls = df.apply(lambda r: classify_kakao(r.이름, r.카테고리, r.그룹), axis=1)
    df = df[cls.notna()].copy()
    df[["유형", "야간접근_현재", "야간접근_조치후", "필요조치"]] = pd.DataFrame(
        [c for c in cls if c is not None], index=df.index)
    df["출처"] = "kakao"
    print(f"[카카오] AED 후보로 분류된 시설 {len(df)}건")
    print(df.유형.value_counts().to_string())

    os.makedirs("outputs", exist_ok=True)
    df.to_json(OUT_JSON, orient="records", force_ascii=False)

    # ── OSM 결과와 병합 ──
    merged = df[["이름", "유형", "야간접근_현재", "야간접근_조치후", "필요조치",
                 "lon", "lat", "출처"]]
    osm_path = "data/output/AED_실좌표_후보_전체POI.csv"
    if os.path.exists(osm_path):
        o = pd.read_csv(osm_path)
        o = o[["이름", "유형", "야간접근_현재", "야간접근_조치후", "필요조치",
               "lon", "lat"]].assign(출처="osm")
        merged = pd.concat([merged, o], ignore_index=True)
        # 30m 안에 같은 유형이면 동일 시설로 보고 카카오를 우선 채택
        merged["_x"] = (merged.lon * 91_000).round(-1)
        merged["_y"] = (merged.lat * 111_000).round(-1)
        merged["_p"] = (merged.출처 == "kakao").astype(int)
        merged = (merged.sort_values("_p", ascending=False)
                        .drop_duplicates(subset=["_x", "_y", "유형"])
                        .drop(columns=["_x", "_y", "_p"]))

    os.makedirs("data/output", exist_ok=True)
    merged.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n[병합] 통합 후보 {len(merged)}건 "
          f"(카카오 {int((merged.출처=='kakao').sum())} · OSM {int((merged.출처=='osm').sum())})")
    print(f"저장: {OUT_JSON}\n      {OUT_CSV}")
    return merged


if __name__ == "__main__":
    main()
