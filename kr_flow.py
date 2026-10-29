"""
[한국판] 수급 체크
- flow_list.txt의 종목(주식·ETF)마다 네이버 증권에서 투자자별 순매수(외국인·기관·개인)를 받아
- 직전 거래일 수급, 1주(5거래일)·2주(10거래일) 누적 수급, 연속 매수·매도 일수, 흐름 변화를 정리해
- 하루 1회 디스코드로 보냅니다.

데이터: 네이버 증권 (① 모바일 API → 실패하면 ② PC 종목 페이지 '외국인·기관' 탭)
금액(억 원) = 순매수 수량 × 그날 종가 (근사치)
"""
import datetime as dt
import io
import json
import os
import re
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

# ───────────── 설정 ─────────────
WEBHOOK = os.environ.get("DISCORD_WEBHOOK_URL", "")
DAYS = int(os.environ.get("FLOW_DAYS", "10"))                 # 2주 = 10거래일
WEEK = int(os.environ.get("FLOW_WEEK", "5"))                  # 1주 = 5거래일
ACCEL = float(os.environ.get("FLOW_ACCEL", "1.5"))            # 최근 1주가 그 전 1주보다 1.5배 이상이면 '가속'
DIVERGE_PCT = float(os.environ.get("FLOW_DIVERGE", "3"))      # 2주 주가 변화 ±3% 이상인데 수급이 반대면 '엇갈림'
# ───────────────────────────────

ROOT = Path(__file__).parent
KST = ZoneInfo("Asia/Seoul")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/126.0 Safari/537.36",
      "Referer": "https://finance.naver.com/"}
WHO = [("frgn", "외국인"), ("inst", "기관"), ("indv", "개인")]


# ───────────── 목록 ─────────────
def load_list() -> list[tuple[str, list[str]]]:
    groups, cur = [], None
    for ln in (ROOT / "flow_list.txt").read_text(encoding="utf-8").splitlines():
        m = re.match(r"\s*#\s*──\s*(.+?)\s*──", ln)
        if m:
            cur = (m.group(1), [])
            groups.append(cur)
            continue
        code = ln.split("#")[0].strip().upper()
        if code:
            if cur is None:
                cur = ("종목", [])
                groups.append(cur)
            cur[1].append(code.split(",")[0].strip())
    return [g for g in groups if g[1]]


# ───────────── 네이버 데이터 ─────────────
def _num(x) -> float:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    s = str(x).replace(",", "").replace("%", "").replace("+", "").strip()
    try:
        return float(s)
    except ValueError:
        return np.nan


def _get(url: str) -> requests.Response | None:
    for _ in range(3):
        try:
            r = requests.get(url, headers=UA, timeout=20)
            if r.status_code == 200:
                return r
            print(f"[경고] {url} → {r.status_code}")
        except Exception as e:
            print(f"[경고] {url} → {e}")
        time.sleep(1.5)
    return None


def fetch_mobile(code: str) -> pd.DataFrame | None:
    """① 모바일 API (외국인·기관·개인 모두 있음)"""
    r = _get(f"https://m.stock.naver.com/api/stock/{code}/trend?pageSize={DAYS + 12}")
    if r is None:
        return None
    try:
        items = r.json()
    except ValueError:
        return None
    if isinstance(items, dict):
        items = next((v for v in items.values() if isinstance(v, list)), [])
    if not items or not isinstance(items[0], dict):
        return None
    keys = items[0].keys()
    def find(*must):
        return next((k for k in keys if all(m.lower() in k.lower() for m in must)), None)
    k_date = find("bizdate") or find("date")
    k_f, k_i, k_p = find("foreigner", "pure"), find("organ", "pure"), find("individual", "pure")
    k_c, k_v, k_r = find("closeprice"), find("tradingvolume") or find("volume"), find("foreigner", "ratio")
    if not (k_date and k_f and k_i and k_c):
        print(f"[경고] {code} 모바일 API 형식이 예상과 다름: {list(keys)[:12]}")
        return None
    rows = []
    for it in items:
        rows.append({"date": pd.to_datetime(str(it[k_date])[:8], format="%Y%m%d", errors="coerce"),
                     "close": _num(it.get(k_c)), "volume": _num(it.get(k_v)) if k_v else np.nan,
                     "frgn": _num(it.get(k_f)), "inst": _num(it.get(k_i)),
                     "indv": _num(it.get(k_p)) if k_p else np.nan,
                     "ratio": _num(it.get(k_r)) if k_r else np.nan})
    df = pd.DataFrame(rows).dropna(subset=["date", "close"]).set_index("date").sort_index()
    return df if len(df) >= 3 else None


def fetch_pc(code: str) -> pd.DataFrame | None:
    """② PC 페이지 '외국인·기관' 탭 (외국인·기관만, 개인 없음)"""
    frames = []
    for page in (1, 2):
        r = _get(f"https://finance.naver.com/item/frgn.naver?code={code}&page={page}")
        if r is None:
            break
        html = r.content.decode("euc-kr", errors="replace")
        try:
            tables = pd.read_html(io.StringIO(html))
        except ValueError:
            break
        for t in tables:
            cols = [" ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in t.columns]
            t.columns = cols
            c_date = next((c for c in cols if "날짜" in c), None)
            c_inst = next((c for c in cols if "기관" in c), None)
            c_frgn = next((c for c in cols if "외국인" in c and "순매매" in c), None)
            if not (c_date and c_inst and c_frgn) or len(t) < 3:
                continue
            c_close = next((c for c in cols if "종가" in c), None)
            c_vol = next((c for c in cols if "거래량" in c), None)
            c_ratio = next((c for c in cols if "보유율" in c), None)
            d = pd.DataFrame({
                "date": pd.to_datetime(t[c_date].astype(str), format="%Y.%m.%d", errors="coerce"),
                "close": t[c_close].map(_num) if c_close else np.nan,
                "volume": t[c_vol].map(_num) if c_vol else np.nan,
                "frgn": t[c_frgn].map(_num), "inst": t[c_inst].map(_num), "indv": np.nan,
                "ratio": t[c_ratio].map(_num) if c_ratio else np.nan,
            }).dropna(subset=["date", "close"])
            frames.append(d)
            break
        if frames and len(pd.concat(frames)) >= DAYS + 2:
            break
        time.sleep(0.5)
    if not frames:
        return None
    df = pd.concat(frames).drop_duplicates("date").set_index("date").sort_index()
    return df if len(df) >= 3 else None


def fetch_name(code: str) -> str:
    r = _get(f"https://m.stock.naver.com/api/stock/{code}/basic")
    if r is not None:
        try:
            j = r.json()
            for k in ("stockName", "itemName", "name"):
                if j.get(k):
                    return str(j[k])
        except ValueError:
            pass
    r = _get(f"https://finance.naver.com/item/main.naver?code={code}")
    if r is not None:
        m = re.search(r"<title>\s*(.+?)\s*:", r.content.decode("euc-kr", errors="replace"))
        if m:
            return m.group(1)
    return code


def fetch(code: str) -> tuple[pd.DataFrame | None, str]:
    df = fetch_mobile(code)
    src = "모바일"
    if df is None:
        df, src = fetch_pc(code), "PC"
    return df, src


# ───────────── 분석 ─────────────
def streak(s: pd.Series) -> int:
    """오늘부터 거꾸로 같은 부호가 이어진 일수 (+면 연속 순매수, -면 연속 순매도)"""
    v = s.dropna().tolist()[::-1]
    if not v or v[0] == 0:
        return 0
    sign, n = np.sign(v[0]), 0
    for x in v:
        if np.sign(x) != sign:
            break
        n += 1
    return int(n * sign)


def analyze(df: pd.DataFrame, name: str) -> dict:
    df = df.tail(DAYS).copy()
    amt = {k: df[k] * df["close"] / 1e8 for k, _ in WHO}                 # 억 원
    turnover = (df["volume"] * df["close"] / 1e8)
    last = df.index[-1]
    r = {"date": last.date(), "close": float(df["close"].iloc[-1]),
         "chg1": (df["close"].iloc[-1] / df["close"].iloc[-2] - 1) * 100 if len(df) > 1 else np.nan,
         "chg10": (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100,
         "has_indv": bool(df["indv"].notna().any())}
    for k, _ in WHO:
        a = amt[k]
        r[f"{k}_1"] = float(a.iloc[-1]) if pd.notna(a.iloc[-1]) else np.nan
        r[f"{k}_5"] = float(a.tail(WEEK).sum(min_count=1))
        r[f"{k}_prev5"] = float(a.iloc[:-WEEK].tail(WEEK).sum(min_count=1)) if len(a) > WEEK else np.nan
        r[f"{k}_10"] = float(a.sum(min_count=1))
        r[f"{k}_streak"] = streak(df[k])
    tv5 = turnover.tail(WEEK).sum(min_count=1)
    r["fi5_share"] = (r["frgn_5"] + r["inst_5"]) / tv5 * 100 if tv5 and pd.notna(tv5) else np.nan
    ratio = df["ratio"].dropna()
    r["ratio"] = float(ratio.iloc[-1]) if len(ratio) else np.nan
    r["ratio_chg"] = float(ratio.iloc[-1] - ratio.iloc[0]) if len(ratio) > 1 else np.nan

    # 흐름 판정 (1주 기준)
    f5, i5 = r["frgn_5"], r["inst_5"]
    if f5 > 0 and i5 > 0:
        r["verdict"] = ("🟢", "외국인·기관 동반 순매수")
    elif f5 > 0 >= i5:
        r["verdict"] = ("🔵", "외국인 주도 매수")
    elif i5 > 0 >= f5:
        r["verdict"] = ("🟣", "기관 주도 매수")
    else:
        r["verdict"] = ("🔴", "외국인·기관 동반 순매도")

    notes = []
    for k, label in WHO[:2]:
        now, prev = r[f"{k}_5"], r[f"{k}_prev5"]
        if pd.notna(prev) and prev != 0 and np.sign(now) != np.sign(prev) and now != 0:
            notes.append(f"{label} {'매도→매수' if now > 0 else '매수→매도'} 전환")
        elif pd.notna(prev) and np.sign(now) == np.sign(prev) and abs(now) >= abs(prev) * ACCEL and abs(now) > 0:
            notes.append(f"{label} {'매수' if now > 0 else '매도'} 가속")
        s = r[f"{k}_streak"]
        if abs(s) >= 3:
            notes.append(f"{label} {abs(s)}일 연속 순{'매수' if s > 0 else '매도'}")
    fi10 = r["frgn_10"] + r["inst_10"]
    if r["chg10"] <= -DIVERGE_PCT and fi10 > 0:
        notes.append("주가 하락 중 외국인·기관 순매수 (저가 매집 가능성)")
    elif r["chg10"] >= DIVERGE_PCT and fi10 < 0:
        notes.append("주가 상승 중 외국인·기관 순매도 (개인이 끌어올림, 주의)")
    if r["has_indv"] and "인버스" in name and r["indv_5"] > 0:
        notes.append("개인이 인버스 순매수 → 개인 하락 베팅 증가")
    if "레버리지" in name and r["has_indv"] and r["indv_5"] > 0:
        notes.append("개인이 레버리지 순매수 → 개인 상승 베팅 증가")
    if pd.notna(r["ratio_chg"]) and abs(r["ratio_chg"]) >= 0.3:
        notes.append(f"외국인 보유율 {r['ratio']:.1f}% (2주 {r['ratio_chg']:+.1f}%p)")
    r["notes"] = notes
    return r


# ───────────── 디스코드 ─────────────
def post(content: str):
    if not WEBHOOK:
        print(content)
        print("-" * 40)
        return
    for _ in range(5):
        r = requests.post(WEBHOOK, json={"content": content}, timeout=30)
        if r.status_code == 429:
            time.sleep(float(r.json().get("retry_after", 2)) + 0.5)
            continue
        r.raise_for_status()
        time.sleep(0.6)
        return


def won(x: float, w: int = 8) -> str:
    """억 원 표시: 10억 이상은 정수, 미만은 소수 1자리"""
    if x is None or pd.isna(x):
        return f"{'-':>{w}}"
    if abs(x) < 0.05:
        s = "0"                                                   # -0.0 / +0.0 대신 0
    else:
        s = f"{x:+,.0f}" if abs(x) >= 10 else f"{x:+.1f}"
    return f"{s:>{w}}"


SEP = "─" * 44                                                    # 종목 카드 사이 구분선


def fit(s: str, w: int, right: bool = False) -> str:
    """디스코드 고정폭 글꼴에서 한글은 두 칸"""
    import unicodedata
    wd = sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in s)
    pad = " " * max(w - wd, 0)
    return pad + s if right else s + pad


def card(code: str, name: str, r: dict) -> list[str]:
    head = f"{name} {code}  종가 {r['close']:,.0f}" + (f" ({r['chg1']:+.1f}%)" if pd.notna(r["chg1"]) else "")
    lines = [head, fit("", 6) + fit("전일", 8, True) + fit("1주", 8, True) + fit("2주", 8, True) + fit("연속", 7, True)]
    for k, label in WHO:
        if k == "indv" and not r["has_indv"]:
            continue
        s = r[f"{k}_streak"]
        st = f"{s:+d}d" if s else "-"
        lines.append(fit(label, 6) + won(r[f"{k}_1"]) + won(r[f"{k}_5"]) + won(r[f"{k}_10"]) + f"{st:>7}")
    e, v = r["verdict"]
    tail = f" · 거래대금 대비 {r['fi5_share']:+.0f}%" if pd.notna(r["fi5_share"]) else ""
    lines.append(f"→ {e} {v} (1주){tail}")
    for n in r["notes"]:
        lines.append(f"  · {n}")
    return lines


def send_cards(title: str, blocks: list[list[str]]):
    buf, first = [], True
    def flush():
        nonlocal buf, first
        if buf:
            post((f"**{title}**\n" if first else "") + "```\n" + "\n".join(buf) + "\n```")
            buf, first = [], False
    for b in blocks:
        if sum(len(x) + 1 for x in buf + b) > 1750:
            flush()
        buf += ([SEP] if buf else []) + b                         # 같은 메시지 안의 카드 사이에 구분선
    flush()


# ───────────── 실행 ─────────────
def main():
    groups = load_list()
    results, missing = {}, []
    for _, codes in groups:
        for code in codes:
            if code in results:
                continue
            df, src = fetch(code)
            if df is None:
                missing.append(code)
                continue
            name = fetch_name(code)
            results[code] = (name, analyze(df, name), src)
            time.sleep(0.4)
    if not results:
        post("⚠️ 수급 체크: 네이버에서 데이터를 받지 못했습니다. 사이트 접속이나 구조 변경 문제일 수 있습니다.")
        return 1

    day = max(r[1]["date"] for r in results.values())
    wd = "월화수목금토일"[day.weekday()]
    # ① 한눈에 보기
    lines = []
    for gname, codes in groups:
        rows = [(c, *results[c]) for c in codes if c in results]
        if not rows:
            continue
        lines.append(f"[{gname}]")
        for c, name, r, _ in rows:
            e, _v = r["verdict"]
            lines.append(f"{e} 외{won(r['frgn_5'], 8)} 기{won(r['inst_5'], 8)}"
                         + (f" 개{won(r['indv_5'], 8)}" if r["has_indv"] else "") + f"  {name}")
        lines.append("")
    post(f"**💰 수급 체크 | {day:%m/%d}({wd}) 기준**\n"
         "1주(5거래일) 누적 순매수, 단위 억 원 · 🟢 외·기 동반 매수 🔵 외국인 주도 🟣 기관 주도 🔴 동반 매도\n"
         "```\n" + "\n".join(lines).strip() + "\n```")

    # ② 묶음별 상세
    for gname, codes in groups:
        blocks = [card(c, results[c][0], results[c][1]) for c in codes if c in results]
        if blocks:
            send_cards(f"📋 {gname}", blocks)

    srcs = {s for _, _, s in results.values()}
    post("단위 억 원 (순매수 수량 × 종가, 근사치) · 1주 = 최근 5거래일 · 2주 = 최근 10거래일 · 연속 = 오늘까지 같은 방향이 이어진 거래일 수 (+4d = 4일 연속 순매수)\n"
         "거래대금 대비 = 1주 외국인+기관 순매수 ÷ 1주 거래대금\n"
         "⚠️ ETF의 기관 수급에는 유동성공급자(LP, 증권사)의 호가 제공 물량이 섞여 있어 주식보다 의미가 약합니다.\n"
         + ("개인 수급은 이번에 받지 못했습니다 (PC 페이지 사용)." if srcs == {"PC"} else ""))
    if missing:
        post("⚠️ 수급을 받지 못한 종목: " + ", ".join(missing) + " (코드 확인 또는 일시적 오류)")
    print(f"완료: {len(results)}종목, 실패 {len(missing)}")


if __name__ == "__main__":
    sys.exit(main())
