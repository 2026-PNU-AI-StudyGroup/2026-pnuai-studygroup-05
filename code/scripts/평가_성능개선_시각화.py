# -*- coding: utf-8 -*-
"""07. 성능평가 시각화 — 형평성·비용효과·표고반전·신규커버 + 전/후 커버 지도

입력: data/output/평가_집계구별.csv (평가_성능개선.py 산출), outputs/oa_risk.parquet
출력: results/평가_종합차트.png, results/평가_커버지도.png
경로: 저장소 루트 기준 상대경로.
"""
import sys, os
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
for _c in [Path.cwd(), *Path.cwd().parents]:
    if (_c / "README.md").exists() and (_c / ".gitignore").exists():
        os.chdir(_c); break

import numpy as np, pandas as pd, geopandas as gpd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
ACC, TEAL, GOLD, GREY = "#c0392b", "#2c7a7b", "#b8860b", "#95a5a6"

ev = pd.read_csv("data/output/평가_집계구별.csv", encoding="utf-8-sig")
Path("results").mkdir(exist_ok=True)

# ── 종합 4패널 ──
fig, ax = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("AED 신규 배치 성능평가 — 기존(106) vs 통합(106+신규 36), 야간 기준",
             fontsize=14, fontweight="bold", y=0.98)

# (1) 형평성
a = ax[0,0]; thrs=[4,6,8]; before=[27.1,19.4,9.1]; after=[79.7,93.5,90.9]
x=np.arange(3); w=0.36
a.bar(x-w/2, before, w, label="기존 106", color=GREY)
a.bar(x+w/2, after, w, label="통합(+신규)", color=ACC)
for i,(b,af) in enumerate(zip(before,after)):
    a.text(i-w/2,b+2,f"{b:.0f}%",ha="center",fontsize=9)
    a.text(i+w/2,af+2,f"{af:.0f}%",ha="center",fontsize=9,fontweight="bold",color=ACC)
    a.annotate(f"+{af-b:.0f}%p",(i,af+8),ha="center",fontsize=9,color=TEAL,fontweight="bold")
a.set_xticks(x); a.set_xticklabels([f"{t}분 초과\n취약지" for t in thrs])
a.set_ylabel("야간 커버율 (%)"); a.set_ylim(0,110)
a.set_title("① 형평성 — 구급차 늦은 곳일수록 크게 개선", fontsize=11, fontweight="bold")
a.legend(loc="upper left"); a.grid(axis="y", alpha=0.3)

# (2) 비용-효과
a = ax[0,1]; stages=["기존\n106","+Tier1\n무비용","+Tier2\n저비용","+Tier3\n고비용"]
cum=[1294,1888,2317,4028]; cols=[GREY,TEAL,GOLD,ACC]
a.plot(range(4),cum,"-o",color="#555",lw=1.5,zorder=1)
for i,(c,col) in enumerate(zip(cum,cols)):
    a.scatter(i,c,s=120,color=col,zorder=2); a.text(i,c+120,f"{c:,}명",ha="center",fontsize=9,fontweight="bold")
a.axvspan(0.5,1.5,alpha=0.12,color=TEAL)
a.text(1,3700,"무비용 구간\n+594명",ha="center",fontsize=9,color=TEAL,fontweight="bold")
a.set_xticks(range(4)); a.set_xticklabels(stages); a.set_ylabel("야간 커버 75세+ 인구 (명)")
a.set_title("② 비용-효과 — 무비용(기존시설)만으로 +594명", fontsize=11, fontweight="bold")
a.grid(axis="y", alpha=0.3)

# (3) 표고 반전
a = ax[1,0]; data=[13,50,66]; labs=["기존 야간AED\n(중앙)","신규 후보\n(중앙)","고위험 집계구\n(중앙)"]
bars=a.bar(labs,data,color=[GREY,ACC,"#7b241c"])
for b,d in zip(bars,data): a.text(b.get_x()+b.get_width()/2,d+1.5,f"{d}m",ha="center",fontweight="bold")
a.axhline(13,ls="--",color=GREY,alpha=0.6)
a.annotate("",xy=(1,50),xytext=(0,13),arrowprops=dict(arrowstyle="->",color=TEAL,lw=2))
a.text(0.5,32,"편중 반전\n평지→고지대",ha="center",fontsize=9,color=TEAL,fontweight="bold")
a.set_ylabel("표고 (m)")
a.set_title("③ 편중 반전 — '필요한 곳 ≠ 설치된 곳' 해소", fontsize=11, fontweight="bold")
a.grid(axis="y", alpha=0.3)

# (4) 신규커버 산점도
a = ax[1,1]
got=ev[ev["신규커버_획득"]==True]
had=ev[(ev["커버_통합야간"]==True)&(ev["신규커버_획득"]==False)]
un=ev[ev["커버_통합야간"]==False]
a.scatter(had["구급차도달_분"],had["elev"],s=had["p75_2026"]/2,color=GREY,alpha=0.5,label="기존이 이미 커버")
a.scatter(got["구급차도달_분"],got["elev"],s=got["p75_2026"]/2,color=ACC,alpha=0.7,label="신규가 새로 커버")
a.scatter(un["구급차도달_분"],un["elev"],s=un["p75_2026"]/2,facecolors="none",edgecolors="#333",label="여전히 미커버")
a.axvline(4,ls="--",color=GOLD,alpha=0.7)
a.set_xlabel("구급차 도달시간 t1+t2 (분)"); a.set_ylabel("집계구 표고 (m)")
a.set_title("④ 신규가 덮은 곳 — 도달 느리고 높은 곳(점크기=75+인구)", fontsize=11, fontweight="bold")
a.legend(loc="upper left", fontsize=8); a.grid(alpha=0.3)

plt.tight_layout(rect=[0,0,1,0.96]); plt.savefig("results/평가_종합차트.png", dpi=130)
print("저장: results/평가_종합차트.png")

# ── 커버 지도 (집계구 경계가 있을 때만; 저장소엔 산출물 미포함) ──
if not Path("outputs/oa_risk.parquet").exists():
    print("커버 지도 생략: outputs/oa_risk.parquet 없음(집계구 경계). 차트만 생성됨.")
    sys.exit(0)
oa = gpd.read_parquet("outputs/oa_risk.parquet")[["TOT_OA_CD","geometry"]].copy()
oa["TOT_OA_CD"]=oa["TOT_OA_CD"].astype(str); ev["TOT_OA_CD"]=ev["TOT_OA_CD"].astype(str)
g = oa.merge(ev, on="TOT_OA_CD").to_crs(4326)
fig2, mx = plt.subplots(1,2,figsize=(14,7))
fig2.suptitle("야간 AED 커버 — 배치 전(기존 106) vs 후(통합)", fontsize=13, fontweight="bold")
for k,(col,title) in enumerate([("커버_기존야간","배치 전 (기존 106곳)"),("커버_통합야간","배치 후 (통합)")]):
    ax_=mx[k]; g.plot(ax=ax_,color="#eee",edgecolor="#bbb",linewidth=0.5)
    g[g[col]].plot(ax=ax_,color=TEAL,edgecolor="white",linewidth=0.5,alpha=0.75)
    g[~g[col]].plot(ax=ax_,color=ACC,edgecolor="white",linewidth=0.5,alpha=0.5)
    ax_.set_title(f"{title}\n커버 {int(g[col].sum())}/78 집계구 · 75+ {g.loc[g[col],'p75_2026'].sum():,.0f}명",fontsize=10)
    ax_.axis("off")
from matplotlib.patches import Patch
fig2.legend(handles=[Patch(color=TEAL,label="야간 커버"),Patch(color=ACC,alpha=0.5,label="야간 공백")],
            loc="lower center", ncol=2)
plt.tight_layout(rect=[0,0.04,1,0.95]); plt.savefig("results/평가_커버지도.png", dpi=130)
print("저장: results/평가_커버지도.png")
