"""
[한국판] 시장 수급 (국장 전체)
① 시장 전체: 코스피·코스닥 투자자별 순매수 (당일·5일·20일 누적, 억 원)
④ 기관 세부: 금융투자·투신·연기금 등 (시장 단위)
② 종목별: 스캔 대상(시총 1조+·거래대금 100억+ 약 200종목 + 관심종목)의 외국인·기관 1주 순매수 상위·하위
③ 동반 매수: 외국인·기관이 함께 사는 종목 (쌍끌이)
⑤ 수급 주도주: 동반 매수 + 50일선 위 + 52주 고점 근처
⑥ 업종별: 외국인·기관 1주·20일 순매수를 약 25개 업종으로 묶어 유입·이탈·전환

데이터: 네이버 증권 (시장 = 투자자별 매매동향 페이지, 종목 = kr_flow.py와 같은 방식), 차트 = 야후 파이낸스
평일 18:30 (확정 수급 반영 후) 하루 1회
"""
import datetime as dt
import io
import os
import sys
import time

import numpy as np
import pandas as pd

import kr_flow as F
import krcommon as K

# ───────────── 설정 ─────────────
F.DAYS = 22                                                   # 종목 수급: 최근 22거래일 받기 (20일 누적용)
TOP_N = int(os.environ.get("MF_TOP", "10"))                  # 순매수·순매도 상위 몇 개
BOTH_TOP = int(os.environ.get("MF_BOTH_TOP", "15"))          # 동반 매수 표시 개수
LEADER_NEAR_HIGH = float(os.environ.get("MF_NEAR_HIGH", "10"))  # 수급 주도주: 52주 고점 -10% 이내
LEADER_TOP = int(os.environ.get("MF_LEADER_TOP", "20"))       # 수급 주도주 표시 개수
# ───────────────────────────────

MARKETS = [("01", "KOSPI"), ("02", "KOSDAQ")]
INV_COLS = [("개인", "개인"), ("외국인", "외국인"), ("기관계", "기관"),
            ("금융투자", "└ 금융투자"), ("투신", "└ 투신"), ("연기금", "└ 연기금"),
            ("보험", "└ 보험"), ("은행", "└ 은행"), ("기타법인", "기타법인")]


# ───────────── ① 시장 전체 ─────────────
def fetch_market(sosok: str) -> pd.DataFrame | None:
    """네이버 '투자자별 매매동향(일별)' 페이지. 단위 억 원"""
    today = dt.datetime.now(K.KST).strftime("%Y%m%d")
    frames = []
    for page in range(1, 4):
        r = F._get(f"https://finance.naver.com/sise/investorDealTrendDay.naver?bizdate={today}&sosok={sosok}&page={page}")
        if r is None:
            break
        try:
            tables = pd.read_html(io.StringIO(r.content.decode("euc-kr", errors="replace")))
        except ValueError:
            break
        for t in tables:
            cols = [" ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in t.columns]
            t.columns = cols
            c_date = next((c for c in cols if "날짜" in c), None)
            if not c_date or not any("외국인" in c for c in cols):
                continue
            d = pd.DataFrame({"date": pd.to_datetime(t[c_date].astype(str), format="%y.%m.%d", errors="coerce")})
            for key, _ in INV_COLS:
                c = next((c for c in cols if key in c), None)
                if c is not None:
                    d[key] = t[c].map(F._num)
            d = d.dropna(subset=["date"])
            if len(d):
                frames.append(d)
            break
        if frames and sum(len(f) for f in frames) >= 22:
            break
        time.sleep(0.5)
    if not frames:
        return None
    df = pd.concat(frames).drop_duplicates("date").set_index("date").sort_index()
    return df if len(df) >= 5 and "외국인" in df else None


def market_block(name: str, df: pd.DataFrame) -> list[str]:
    lines = [f"{name}  ({df.index[-1]:%m/%d})",
             F.fit("", 12) + F.fit("당일", 9, True) + F.fit("5일", 9, True) + F.fit("20일", 9, True) + F.fit("연속", 7, True)]
    for key, label in INV_COLS:
        if key not in df or df[key].isna().all():
            continue
        s = df[key]
        st = F.streak(s)
        lines.append(F.fit(label, 12) + F.won(s.iloc[-1], 9) + F.won(s.tail(5).sum(), 9) + F.won(s.tail(20).sum(), 9)
                     + f"{(f'{st:+d}d' if st else '-'):>7}")
    notes = []
    for key, label in (("외국인", "외국인"), ("기관계", "기관"), ("연기금", "연기금"), ("개인", "개인")):
        if key in df:
            st = F.streak(df[key])
            if abs(st) >= 3:
                notes.append(f"{label} {abs(st)}일 연속 순{'매수' if st > 0 else '매도'}")
    if "외국인" in df and "기관계" in df:
        f5, i5 = df["외국인"].tail(5).sum(), df["기관계"].tail(5).sum()
        if f5 > 0 and i5 > 0:
            notes.append("최근 1주 외국인·기관 동반 순매수 (수급 우호)")
        elif f5 < 0 and i5 < 0:
            notes.append("최근 1주 외국인·기관 동반 순매도 (개인만 매수, 주의)")
    lines += [f"  · {n}" for n in notes]
    return lines


# ───────────── ② ③ ⑤ 종목별 ─────────────
def stock_flows(universe: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for t, name in zip(universe["ticker"], universe["name"]):
        df, _ = F.fetch(t)
        if df is None or len(df) < 6:
            continue
        df = df.tail(20)
        amt = {k: df[k] * df["close"] / 1e8 for k in ("frgn", "inst")}
        tv5 = (df["volume"] * df["close"]).tail(5).sum() / 1e8
        rows.append({"t": t, "name": name, "close": df["close"].iloc[-1], "date": df.index[-1],
                     "f1": amt["frgn"].iloc[-1], "i1": amt["inst"].iloc[-1],
                     "f5": amt["frgn"].tail(5).sum(), "i5": amt["inst"].tail(5).sum(),
                     "f20": amt["frgn"].sum(), "i20": amt["inst"].sum(),
                     "fs": F.streak(df["frgn"]), "is": F.streak(df["inst"]),
                     "share": (amt["frgn"].tail(5).sum() + amt["inst"].tail(5).sum()) / tv5 * 100 if tv5 else np.nan})
        time.sleep(0.3)
    return pd.DataFrame(rows)


def leaders(both: pd.DataFrame) -> pd.DataFrame:
    """동반 매수 종목 중 50일선 위 + 52주 고점 -10% 이내 (+ 50일선 > 200일선이면 ★)"""
    if both.empty:
        return both
    dl = K.make_download("1y", True)
    keep = []
    for t, px in dl(both["t"].tolist()):
        c = px["Close"].dropna()
        if len(c) < 60:
            continue
        s50 = c.rolling(50).mean().iloc[-1]
        s200 = c.rolling(200).mean().iloc[-1] if len(c) >= 200 else np.nan
        hi = px["High"].tail(252).max()
        from_hi = (c.iloc[-1] / hi - 1) * 100
        if c.iloc[-1] > s50 and from_hi >= -LEADER_NEAR_HIGH:
            keep.append({"t": t, "from_hi": from_hi, "above50": (c.iloc[-1] / s50 - 1) * 100,
                         "star": bool(pd.notna(s200) and s50 > s200)})
    if not keep:
        return pd.DataFrame()
    return both.merge(pd.DataFrame(keep), on="t")


def stock_table(title: str, df: pd.DataFrame, cols: list[tuple[str, str, int]], extra=None):
    if df.empty:
        return
    head = F.fit("CODE", 7) + "".join(F.fit(h, w, True) for h, _, w in cols) + "  종목"
    lines = []
    for _, r in df.iterrows():
        cells = []
        for _, key, w in cols:
            v = r[key]
            if key in ("fs", "is"):
                cells.append(f"{(f'{int(v):+d}d' if v else '-'):>{w}}")
            elif key in ("share", "from_hi", "above50"):
                cells.append(f"{v:>+{w}.0f}" if pd.notna(v) else f"{'-':>{w}}")
            else:
                cells.append(F.won(v, w))
        lines.append(f"{r['t']:<7}" + "".join(cells) + f"  {(extra(r) if extra else '')}{r['name'][:12]}")
    F.send_cards(title, [[head] + lines])



# ───────────── ⑥ 업종 분류 ─────────────
DESC_URL = "https://raw.githubusercontent.com/FinanceData/fdr_krx_data_cache/refs/heads/master/data/listing/desc/{d}.csv"
OVERRIDE = {"005930": "반도체", "000660": "반도체", "066570": "전자/가전"}
# 세부업종만으로는 잘못 묶이는 대형주 (종목명 기준)
OVERRIDE_NAME = {"KB금융": "은행", "하나금융지주": "은행", "한국금융지주": "증권", "메리츠금융지주": "보험",
                 "HD한국조선해양": "조선", "에코프로": "2차전지", "OCI홀딩스": "화학", "한화시스템": "방산/우주",
                 "두산": "지주사", "코웨이": "유통/상사", "삼성물산": "지주사",
                 "삼성전기": "IT부품/전자", "심텍": "IT부품/전자", "동진쎄미켐": "반도체 장비/소재",
                 "솔브레인": "반도체 장비/소재", "ISC": "반도체 장비/소재", "한국타이어앤테크놀로지": "자동차",
                 "LS에코에너지": "전력기기/전선", "HD현대마린솔루션": "조선"}
# (업종명, KRX 세부업종에 들어 있는 글자, 주요제품에 들어 있는 글자) — 위에서부터 먼저 맞는 것
RULES = [
    ("반도체", ["반도체 제조"], []),
    ("로봇", ["특수 목적용 기계", "일반 목적용 기계"], ["로봇"]),
    ("반도체 장비/소재", ["특수 목적용 기계", "측정, 시험", "기타 비금속 광물", "전문, 과학 및 기술 서비스"], []),
    ("반도체 장비/소재", ["전자부품"], ["반도체", "소켓", "프로브"]),
    ("2차전지", ["일차전지 및 이차전지"], []),
    ("2차전지", ["기초 화학물질", "기타 화학제품"], ["양극재", "음극재", "전해액", "분리막", "동박"]),
    ("화장품", ["기타 화학제품"], ["화장품"]),
    ("화학", ["화학", "고무", "플라스틱"], []),
    ("IT부품/전자", ["전자부품", "통신 및 방송 장비", "영상 및 음향", "컴퓨터 및 주변"], []),
    ("전력기기/전선", ["전동기, 발전기", "절연선", "기타 전기장비", "전기 변환"], []),
    ("조선", ["선박 및 보트"], []),
    ("방산/우주", ["항공기,우주선", "무기 및 총포탄", "그외 기타 운송장비"], []),
    ("자동차", ["자동차", "고무제품"], []),
    ("기계/원전", ["일반 목적용 기계", "구조용 금속제품"], []),
    ("철강/비철", ["1차 철강", "1차 비철금속", "금속"], []),
    ("정유/에너지", ["석유 정제", "가스"], []),
    ("바이오/제약", ["의약", "자연과학 및 공학 연구개발", "의료용"], []),
    ("은행", ["은행 및 저축"], []),
    ("은행", ["기타 금융업"], ["은행", "금융지주"]),
    ("증권", ["금융 지원 서비스", "증권"], []),
    ("보험", ["보험"], []),
    ("지주사", ["기타 금융업"], []),
    ("인터넷/SW", ["소프트웨어", "컴퓨터 프로그래밍", "자료처리", "포털"], []),
    ("통신", ["전기 통신", "통신업"], []),
    ("엔터/미디어", ["오디오물", "영화", "방송업", "광고"], []),
    ("건설", ["건설", "건축기술"], []),
    ("운송", ["운송", "창고"], []),
    ("유통/상사", ["소매", "도매", "상품 중개"], []),
    ("음식료/담배", ["식품", "음료", "담배", "도축"], []),
    ("유틸리티", ["전기업", "증기"], []),
]


def load_sectors() -> dict[str, str]:
    """KRX 상장사 설명 데이터(세부업종·주요제품) → 약 25개 업종"""
    today = dt.datetime.now(K.KST).date()
    df = None
    for back in range(0, 15):
        d = today - dt.timedelta(days=back)
        r = F._get(DESC_URL.format(d=d.isoformat()))
        if r is not None:
            try:
                df = pd.read_csv(io.StringIO(r.content.decode("utf-8-sig")), index_col=0, dtype={"Code": str})
                break
            except Exception:
                df = None
    if df is None:
        return {}
    out = {}
    for code, nm, ind, prod in zip(df["Code"], df["Name"], df["Industry"].fillna(""), df["Products"].fillna("")):
        sec = OVERRIDE.get(code) or OVERRIDE_NAME.get(str(nm))
        if not sec:
            for name, inds, prods in RULES:
                if any(k in ind for k in inds) and (not prods or any(k in prod for k in prods)):
                    sec = name
                    break
        out[code] = sec or "기타"
    return out


def sector_table(sf: pd.DataFrame, sectors: dict[str, str]) -> list[str]:
    d = sf.assign(sector=sf["t"].map(sectors).fillna("기타"))
    g = d.groupby("sector").agg(n=("t", "size"), f5=("f5", "sum"), i5=("i5", "sum"),
                                f20=("f20", "sum"), i20=("i20", "sum"))
    g["tot5"] = g["f5"] + g["i5"]
    g["tot20"] = g["f20"] + g["i20"]
    g = g.sort_values("tot5", ascending=False)
    lines = [F.fit("업종", 18) + F.fit("수", 3, True) + F.fit("외1주", 8, True) + F.fit("기1주", 8, True)
             + F.fit("합1주", 8, True) + F.fit("합20일", 9, True)]
    for sec, r in g.iterrows():
        mark = "🟢" if r["f5"] > 0 and r["i5"] > 0 else ("🔴" if r["f5"] < 0 and r["i5"] < 0 else "⚪")
        lines.append(mark + F.fit(sec, 16) + f"{int(r['n']):>3}" + F.won(r["f5"]) + F.won(r["i5"])
                     + F.won(r["tot5"]) + F.won(r["tot20"], 9))
    notes = []
    top = g[g["tot5"] > 0].head(3).index.tolist()
    bot = g[g["tot5"] < 0].tail(3).index.tolist()[::-1]
    if top:
        notes.append("1주 자금 유입 상위: " + ", ".join(top))
    if bot:
        notes.append("1주 자금 이탈 상위: " + ", ".join(bot))
    turn_in = g[(g["tot5"] > 0) & (g["tot20"] < 0)].index.tolist()
    turn_out = g[(g["tot5"] < 0) & (g["tot20"] > 0)].index.tolist()
    if turn_in:
        notes.append("20일은 순매도인데 최근 1주 순매수 (유입 전환): " + ", ".join(turn_in))
    if turn_out:
        notes.append("20일은 순매수인데 최근 1주 순매도 (이탈 전환): " + ", ".join(turn_out))
    return lines + [""] + [f"· {n}" for n in notes]


# ───────────── 실행 ─────────────
def main():
    # ① 시장 전체
    mkts = {}
    for sosok, name in MARKETS:
        df = fetch_market(sosok)
        if df is not None:
            mkts[name] = df
    day = max((d.index[-1] for d in mkts.values()), default=None)

    # ② 종목별
    uni = pd.concat([K.load_universe("kospi"), K.load_universe("kosdaq"), K.load_universe("watchlist")])
    uni = uni.drop_duplicates("ticker")
    sf = stock_flows(uni)
    if sf.empty and not mkts:
        F.post("⚠️ 시장 수급: 네이버에서 데이터를 받지 못했습니다. 사이트 접속이나 구조 변경 문제일 수 있습니다.")
        return 1
    if day is None and not sf.empty:
        day = sf["date"].max()
    wd = "월화수목금토일"[day.weekday()]

    blocks = [market_block(n, d) for n, d in mkts.items()]
    if blocks:
        F.send_cards(f"🏦 시장 수급 | {day:%m/%d}({wd}) 기준 · 단위 억 원", blocks)
    else:
        F.post(f"**🏦 시장 수급 | {day:%m/%d}({wd})**\n⚠️ 시장 전체 투자자별 매매동향을 받지 못했습니다. 종목별 수급만 보냅니다.")

    if not sf.empty:
        sf = sf[sf["date"] == sf["date"].max()]
        flow_cols = [("외1주", "f5", 8), ("기1주", "i5", 8), ("외연속", "fs", 7), ("기연속", "is", 7), ("비중%", "share", 6)]

        # ⑥ 업종별
        sectors = load_sectors()
        if sectors:
            F.send_cards(f"🏭 업종별 수급 (외국인+기관, 스캔 대상 {len(sf)}종목 합계) · 🟢 동반 유입 🔴 동반 이탈",
                         [sector_table(sf, sectors)])
        else:
            F.post("⚠️ 업종 분류 데이터를 받지 못해 업종별 수급은 건너뜁니다.")

        # ③ 동반 매수
        both = sf[(sf["f5"] > 0) & (sf["i5"] > 0)].assign(tot=lambda d: d["f5"] + d["i5"]).sort_values("tot", ascending=False)
        stock_table(f"🤝 외국인·기관 동반 순매수 (1주) · {len(both)}종목 중 상위 {min(BOTH_TOP, len(both))}",
                    both.head(BOTH_TOP), flow_cols)

        # ⑤ 수급 주도주
        lead = leaders(both)
        if not lead.empty:
            lead = lead.sort_values("tot", ascending=False)
            stock_table(f"🚀 수급 주도주 · 동반 순매수 + 50일선 위 + 52주 고점 -{LEADER_NEAR_HIGH:.0f}% 이내 "
                        f"({len(lead)}종목 중 상위 {min(LEADER_TOP, len(lead))})", lead.head(LEADER_TOP),
                        [("외1주", "f5", 8), ("기1주", "i5", 8), ("고점%", "from_hi", 7), ("50일%", "above50", 7)],
                        extra=lambda r: "★" if r["star"] else " ")

        # ② 외국인·기관 상위·하위
        for key, label in (("f5", "외국인"), ("i5", "기관")):
            stock_table(f"🔵 {label} 1주 순매수 TOP {TOP_N}" if key == "f5" else f"🟣 {label} 1주 순매수 TOP {TOP_N}",
                        sf.sort_values(key, ascending=False).head(TOP_N), flow_cols)
            stock_table(f"🔻 {label} 1주 순매도 TOP {TOP_N}",
                        sf.sort_values(key).head(TOP_N), flow_cols)

    F.post("단위 억 원 (종목은 순매수 수량 × 종가, 근사치) · 1주 = 5거래일 · 연속 = 오늘까지 같은 방향 거래일 수 (+4d = 4일 연속 순매수)\n"
           "비중% = 1주 외국인+기관 순매수 ÷ 1주 거래대금 · 수급 주도주 ★ = 50일선 > 200일선 (정배열)\n"
           f"종목 대상: 시총 1조+·거래대금 100억+ 약 {len(uni)}종목 (관심종목 포함), 받은 종목 {len(sf)}개")
    print(f"완료: 시장 {list(mkts)}, 종목 {len(sf)}")


if __name__ == "__main__":
    sys.exit(main())
