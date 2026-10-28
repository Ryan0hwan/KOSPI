"""
[한국판] 미국판 vcp.py을 한국 주식(코스피·코스닥 거래대금 상위 + kr_watchlist.txt)용으로 바꾼 것.
종목 목록·시세·종목명은 krcommon.py, 나머지 판정 로직은 미국판과 같습니다.

마크 미너비니 검색기 (트렌드 템플릿 + VCP 베이스)
1. 52주 고가 대비 -25% 이내
2. 52주 저가 대비 +25% 이상
3. 50일 > 150일 > 200일선 정배열
4. 주가가 150일·200일선 위
5. 200일선이 최소 1개월 이상 상승 (몇 개월째 상승 중인지 표시)
6. 베이스 = VCP(변동성 수축): 눌림 폭이 점점 줄고(예 25%→15%→8%→3%), 저점은 높아지고,
   거래량이 마르는 구간. 가로 박스뿐 아니라 삼각수렴도 포함.
7. 상대강도(RS) 90 이상 (IBD 방식, 스캔 대상 종목 안에서 백분위)

알림: 🚀 피벗 돌파 / 🎯 피벗 근접 VCP 셋업 / 📋 템플릿만 통과한 종목
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
GROUPS = os.environ.get("SCAN_GROUPS", "kospi,kosdaq,watchlist").split(",")
HIGH_DOWN_PCT = float(os.environ.get("HIGH_DOWN_PCT", "25"))    # 1. 52주 고가 대비 최대 하락률(%)
LOW_UP_PCT = float(os.environ.get("LOW_UP_PCT", "25"))          # 2. 52주 저가 대비 최소 상승률(%)
SLOPE_DAYS = int(os.environ.get("SLOPE_DAYS", "21"))            # 5. 200일선 상승 확인 기간(거래일, 약 1개월)
RS_MIN = float(os.environ.get("RS_MIN", "90"))                  # 7. RS 하한
MIN_PRICE = float(os.environ.get("MIN_PRICE", "1000"))
MIN_DOLLAR_VOL = float(os.environ.get("MIN_DOLLAR_VOL", "2000000000"))  # 50일 평균 거래대금(원)

# 6. VCP 베이스
BASE_MAX_DAYS = int(os.environ.get("BASE_MAX_DAYS", "180"))     # 베이스 시작 고점을 찾는 최대 기간(거래일, 약 9개월)
BASE_MIN_DAYS = int(os.environ.get("BASE_MIN_DAYS", "15"))      # 베이스 최소 길이(거래일, 약 3주)
ZZ_PCT = float(os.environ.get("ZZ_PCT", "3"))                   # 고점·저점(스윙)으로 인정할 최소 반전폭(%)
MIN_CONTRACTIONS = int(os.environ.get("MIN_CONTRACTIONS", "2")) # 최소 수축 횟수 (T2 이상)
FIRST_MAX = float(os.environ.get("FIRST_MAX", "40"))            # 첫 수축(가장 깊은 눌림) 최대 깊이(%)
LAST_MAX = float(os.environ.get("LAST_MAX", "12"))              # 마지막 수축 최대 깊이(%)
LAST_RATIO = float(os.environ.get("LAST_RATIO", "0.6"))         # 마지막 수축 ≤ 첫 수축 × 0.6
VDU_MAX = float(os.environ.get("VDU_MAX", "0.9"))               # 최근 10일 평균 거래량 ≤ 50일 평균 × 0.9 (거래량 마름)
PIVOT_NEAR = float(os.environ.get("PIVOT_NEAR", "8"))           # 피벗 아래 8% 이내면 '피벗 근접'
BREAKOUT_DAYS = int(os.environ.get("BREAKOUT_DAYS", "3"))       # 최근 3거래일 안에 피벗을 넘었으면 '돌파'
BREAKOUT_VOL = float(os.environ.get("BREAKOUT_VOL", "1.4"))     # 돌파일 거래량 ≥ 50일 평균 × 1.4 이면 '거래량 동반'
REALERT_DAYS = int(os.environ.get("REALERT_DAYS", "10"))        # 같은 종목 돌파 알림은 10일에 한 번
FORM_LAST_MAX = float(os.environ.get("FORM_LAST_MAX", "20"))    # ⏳ 형성 중: 모양은 VCP인데 마지막 수축이 20% 이하로 아직 덜 조여짐
FORM_NEAR = float(os.environ.get("FORM_NEAR", "15"))            # ⏳ 형성 중: 피벗 아래 15% 이내

WEBHOOK = os.environ.get("DISCORD_WEBHOOK_URL", "")
FORCE = os.environ.get("FORCE", "0") == "1"
# 진단: "ZETA" 또는 "ZETA@2026-01-05" (날짜를 붙이면 그날 기준으로 판정)
DIAGNOSE = []
for _x in os.environ.get("DIAGNOSE", "").replace(" ", "").split(","):
    if _x:
        _t, _, _d = _x.upper().partition("@")
        DIAGNOSE.append((_t, dt.date.fromisoformat(_d) if _d else None))
# ───────────────────────────────

NY = ZoneInfo("America/New_York")
ROOT = Path(__file__).parent
STATE = ROOT / "state" / "vcp"
STATE.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (vcp-scanner)"}


# ───────────── 종목 목록 ─────────────
def _fetch_sp500() -> pd.DataFrame:
    html = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", headers=UA, timeout=30).text
    t = pd.read_html(io.StringIO(html))[0]
    return pd.DataFrame({"ticker": t["Symbol"], "name": t["Security"], "industry": t["GICS Sub-Industry"]})


def _load_watchlist() -> pd.DataFrame:
    rows = []
    p = ROOT / "watchlist.txt"
    if p.exists():
        for ln in p.read_text(encoding="utf-8").splitlines():
            ln = ln.split("#")[0].strip()
            if ln:
                tk, _, name = ln.partition(",")
                tk = re.sub(r"\.([A-Z])$", r"-\1", tk.strip().upper())
                rows.append({"ticker": tk, "name": name.strip(), "industry": "Watchlist"})
    return pd.DataFrame(rows, columns=["ticker", "name", "industry"]).drop_duplicates("ticker")


def load_universe(group: str) -> pd.DataFrame:
    if group == "watchlist":
        return _load_watchlist()
    cache = STATE / f"universe_{group}.csv"
    if not (cache.exists() and time.time() - cache.stat().st_mtime < 7 * 86400):
        try:
            df = _fetch_sp500()
            if len(df) < 400:
                raise ValueError("종목 수가 비정상적으로 적음")
            df.to_csv(cache, index=False)
        except Exception as e:
            print(f"[경고] {group} 목록 갱신 실패: {e}")
    if not cache.exists():
        return pd.DataFrame(columns=["ticker", "name", "industry"])
    df = pd.read_csv(cache).fillna("")
    df["ticker"] = df["ticker"].astype(str).str.strip().str.replace(".", "-", regex=False)
    return df[df["ticker"].str.fullmatch(r"[A-Z\-]{1,6}")].drop_duplicates("ticker")


def download(tickers: list[str]):
    import yfinance as yf
    for i in range(0, len(tickers), 200):
        chunk = tickers[i:i + 200]
        try:
            data = yf.download(chunk, period="2y", interval="1d", group_by="ticker",
                               auto_adjust=True, threads=True, progress=False)
        except Exception as e:
            print(f"[경고] 시세 다운로드 실패({chunk[0]}…): {e}")
            continue
        if not isinstance(data.columns, pd.MultiIndex):
            data = pd.concat({chunk[0]: data}, axis=1)
        have = set(data.columns.get_level_values(0))
        for t in chunk:
            if t in have:
                yield t, data[t].dropna(subset=["Close", "High", "Low", "Volume"])
        time.sleep(1)


# ───────────── 트렌드 템플릿 (조건 1~5, 7) ─────────────
def rs_raw(c: pd.Series) -> float:
    """IBD 방식 가중 수익률: 3개월 ×2 + 6·9·12개월 (있는 기간만)"""
    parts = [(63, 2), (126, 1), (189, 1), (252, 1)]
    num = den = 0.0
    for n, w in parts:
        if len(c) > n:
            num += (c.iloc[-1] / c.iloc[-1 - n] - 1) * w
            den += w
    return num / den if den else np.nan


def template(df: pd.DataFrame) -> dict | None:
    if len(df) < 200 + SLOPE_DAYS + 1:
        return None
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    s50, s150, s200 = (c.rolling(n).mean() for n in (50, 150, 200))
    hi52, lo52 = h.tail(252).max(), l.tail(252).min()
    last = c.iloc[-1]
    # 200일선이 몇 개월째 상승 중인지 (매일 '21일 전보다 높음'이 연속된 일수 ÷ 21)
    up = (s200 > s200.shift(SLOPE_DAYS)).dropna()
    streak = 0
    for ok in reversed(up.tolist()):
        if not ok:
            break
        streak += 1
    r = {
        "c1": last >= hi52 * (1 - HIGH_DOWN_PCT / 100),
        "c2": last >= lo52 * (1 + LOW_UP_PCT / 100),
        "c3": s50.iloc[-1] > s150.iloc[-1] > s200.iloc[-1],
        "c4": last > s150.iloc[-1] and last > s200.iloc[-1],
        "c5": bool(up.iloc[-1]) if len(up) else False,
        "from_hi": (last / hi52 - 1) * 100, "from_lo": (last / lo52 - 1) * 100,
        "rising_m": (streak + SLOPE_DAYS) / 21 if streak else 0.0,
        "s50": s50.iloc[-1], "s150": s150.iloc[-1], "s200": s200.iloc[-1], "close": last,
        "liquid": last >= MIN_PRICE and (c * v).tail(50).mean() >= MIN_DOLLAR_VOL,
        "rs_raw": rs_raw(c),
    }
    return r


# ───────────── 6. VCP 베이스 감지 ─────────────
def zigzag(h: np.ndarray, l: np.ndarray, start: int, pct: float):
    """start(베이스 시작 고점)부터 고점·저점 스윙을 찾음.
    반환: pivots [(idx, price, 'H'|'L')], 마지막 상태('down' = 고점 이후 하락 진행 중, 'up' = 저점 이후 상승 중), 진행 중 극값"""
    th = pct / 100
    piv = [(start, h[start], "H")]
    mode, ext_i, ext_p = "down", start, l[start]
    for i in range(start + 1, len(h)):
        if mode == "down":
            if l[i] < ext_p:
                ext_i, ext_p = i, l[i]
            if h[i] >= ext_p * (1 + th):                 # 저점 확정 → 상승 전환
                piv.append((ext_i, ext_p, "L"))
                mode, ext_i, ext_p = "up", i, h[i]
        else:
            if h[i] > ext_p:
                ext_i, ext_p = i, h[i]
            if l[i] <= ext_p * (1 - th):                 # 고점 확정 → 하락 전환
                piv.append((ext_i, ext_p, "H"))
                mode, ext_i, ext_p = "down", i, l[i]
    return piv, mode, (ext_i, ext_p)


def find_vcp(df: pd.DataFrame) -> dict:
    """오늘 기준 VCP 판정. 반환 dict의 ok=True면 통과, 아니면 why에 탈락 사유"""
    out = {"ok": False, "why": ""}
    n = len(df)
    if n < BASE_MAX_DAYS + 5:
        out.update(why="데이터 부족", code="data")
        return out
    h, l, c, v = (df[k].to_numpy(float) for k in ("High", "Low", "Close", "Volume"))
    lo = n - BASE_MAX_DAYS
    start = lo + int(np.argmax(h[lo:]))                  # 베이스 시작 = 최근 9개월 최고가
    base_len = n - 1 - start
    out.update(start=start, base_len=base_len)
    if base_len < BASE_MIN_DAYS:
        out.update(why=f"베이스가 너무 짧음 ({base_len}일, 최근 고점에서 아직 베이스를 만들지 않음)", code="short")
        return out

    piv, mode, (ei, ep) = zigzag(h, l, start, ZZ_PCT)
    # 수축 목록: (고점, 그 다음 저점)
    cons = []
    for a, b in zip(piv, piv[1:]):
        if a[2] == "H" and b[2] == "L":
            cons.append({"hi_i": a[0], "hi": a[1], "lo_i": b[0], "lo": b[1]})
    last_h = [p for p in piv if p[2] == "H"][-1]
    if mode == "down" and ei > last_h[0]:                # 마지막 고점 이후 하락 진행 중 → 진행 중인 수축
        cons.append({"hi_i": last_h[0], "hi": last_h[1], "lo_i": ei, "lo": ep, "open": True})
    for k in cons:
        k["depth"] = (k["hi"] - k["lo"]) / k["hi"] * 100
    out["cons"] = cons
    if len(cons) < MIN_CONTRACTIONS:
        out.update(why=f"수축 횟수 부족 ({len(cons)}회 < {MIN_CONTRACTIONS}회)", code="few")
        return out
    d = [k["depth"] for k in cons]
    d1, dl = d[0], d[-1]
    # 감소하는 수축 개수 (첫 수축 → 마지막 수축으로 이어지는 가장 긴 감소 수열 = T)
    best = [1] * len(d)
    for i in range(1, len(d)):
        for j in range(i):
            if d[i] < d[j] and best[j] + 1 > best[i]:
                best[i] = best[j] + 1
    T = best[-1]
    out.update(T=T, depths=d)
    half = len(d) // 2
    # (통과 여부, 사유, 코드, 구조 조건인지) — 구조 조건은 VCP 모양 자체, 나머지는 '조여짐·거래량'
    checks = [
        (d1 <= FIRST_MAX, f"첫 수축이 너무 깊음 ({d1:.0f}% > {FIRST_MAX:.0f}%)", "first_deep", True),
        (max(d) <= d1 * 1.1, f"중간에 첫 수축보다 깊은 눌림이 있음 (최대 {max(d):.0f}%)", "expand", True),
        (cons[-1]["lo"] >= cons[0]["lo"], "저점이 높아지지 않음 (마지막 저점 < 첫 저점)", "lows", True),
        (dl <= LAST_MAX, f"마지막 수축이 아직 큼 ({dl:.0f}% > {LAST_MAX:.0f}%)", "last_big", False),
        (dl <= d1 * LAST_RATIO, f"수축이 충분히 줄지 않음 (마지막 {dl:.0f}% / 첫 {d1:.0f}%)", "last_big", False),
        (np.mean(d[half:]) < np.mean(d[:half]) if half else True, "뒤쪽 수축이 앞쪽보다 작지 않음", "last_big", False),
        (T >= MIN_CONTRACTIONS, f"점점 줄어드는 수축이 {T}회뿐", "last_big", False),
    ]
    # 거래량 마름: 최근 10일 평균 ≤ 50일 평균 × VDU_MAX, 마지막 수축 구간 평균 < 첫 수축 구간 평균
    avg50 = v[-51:-1].mean()
    vdu = v[-10:].mean() / avg50 if avg50 else np.nan
    first_leg = v[cons[0]["hi_i"]:cons[0]["lo_i"] + 1].mean()
    last_leg = v[cons[-1]["hi_i"]:].mean()
    out.update(vdu=vdu * 100, leg_ratio=last_leg / first_leg * 100 if first_leg else np.nan)
    checks += [(vdu <= VDU_MAX, f"최근 거래량이 마르지 않음 (10일/50일 = {vdu * 100:.0f}%)", "vdu", False),
               (last_leg < first_leg, f"마지막 수축 구간 거래량이 첫 수축보다 많음 ({out['leg_ratio']:.0f}%)", "vdu", False)]
    pivot = cons[-1]["hi"]                              # 피벗 = 마지막 수축이 시작된 고점
    out.update(pivot=pivot, to_pivot=(pivot / c[-1] - 1) * 100)
    fails = [(why, code, structural) for ok, why, code, structural in checks if not ok]
    if fails:
        out.update(why=fails[0][0], code=fails[0][1], todo=sorted({f[1] for f in fails}))
        # ⏳ 형성 중: 모양(구조 조건)은 VCP이고, 마지막 수축이 아직 덜 조여졌거나 거래량이 덜 마른 상태
        out["forming"] = (not any(f[2] for f in fails) and dl <= FORM_LAST_MAX
                          and out["to_pivot"] <= FORM_NEAR and c[-1] <= pivot)
        return out
    out["ok"] = True
    return out


def check_vcp(df: pd.DataFrame) -> dict:
    """오늘 VCP 셋업인지, 최근 BREAKOUT_DAYS일 안에 피벗을 돌파했는지.
    돌파 후에는 가격이 베이스 위로 올라가 베이스가 '끝난' 상태이므로, 돌파 전날까지의 데이터로 베이스를 찾음"""
    c, v = df["Close"].to_numpy(float), df["Volume"].to_numpy(float)
    first = None
    for k in range(0, BREAKOUT_DAYS + 1):
        r = find_vcp(df.iloc[:len(df) - k] if k else df)
        if k == 0:
            first = r
        if not r["ok"]:
            continue
        p = r["pivot"]
        if k == 0 and c[-1] <= p:                        # 아직 피벗 아래
            if r["to_pivot"] <= PIVOT_NEAR:
                r["status"] = "setup"
            else:
                r.update(status="far", ok=False, code="far", why=f"피벗까지 아직 멂 ({r['to_pivot']:.1f}% > {PIVOT_NEAR:.0f}%)")
            return r
        after = c[len(c) - k:] if k else c[-1:]          # 베이스 이후 종가들
        if (after > p).any() and c[-1] > p * 0.97:      # 피벗을 넘었고, 피벗 -3% 아래로 다시 무너지지 않음
            bi = len(c) - len(after) + int(np.argmax(after > p))
            avg50 = v[bi - 50:bi].mean()
            r.update(status="breakout", bo_days=len(c) - 1 - bi,
                     bo_vol=v[bi] / avg50 * 100 if avg50 else np.nan, bo_pct=(c[-1] / p - 1) * 100)
            return r
    if first.get("forming"):
        first["status"] = "forming"
    return first


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


def send_block(title: str, head: str, lines: list[str]):
    first = True
    while lines:
        block, size = [], 0
        while lines and size + len(lines[0]) + 1 < 1700:
            size += len(lines[0]) + 1
            block.append(lines.pop(0))
        header = f"**{title}**\n" if first else ""
        post(f"{header}```\n{head}\n" + "\n".join(block) + "\n```")
        first = False


def fit(s: str, w: int, right: bool = False) -> str:
    """디스코드 고정폭 글꼴에서 한글은 두 칸 → 표시 너비 기준으로 맞춤"""
    import unicodedata
    wd = sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in s)
    pad = " " * max(w - wd, 0)
    return pad + s if right else s + pad


def depths_str(d: list[float]) -> str:
    s = "/".join(f"{x:.0f}" for x in d[-5:])
    return ("…" + s) if len(d) > 5 else s


def company_name(ticker: str, meta: dict) -> str:
    m = meta.get(ticker)
    name = str(getattr(m, "name", "") or "") if m is not None else ""
    if name:
        return name
    cache_file = STATE / "names.json"
    try:
        cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    except Exception:
        cache = {}
    if ticker not in cache:
        try:
            import yfinance as yf
            info = yf.Ticker(ticker).info
            cache[ticker] = info.get("shortName") or info.get("longName") or ""
        except Exception:
            cache[ticker] = ""
        cache_file.write_text(json.dumps(cache))
    return cache[ticker]


def diagnose(t: str, df: pd.DataFrame | None, tp: dict | None, rs: float | None):
    if df is None or tp is None:
        post(f"🔍 **{t}** 진단: 시세가 없거나 데이터가 부족합니다 (상장 약 1년 미만 등).")
        return
    mark = lambda ok: "✅" if ok else "❌"
    lines = [
        f"{mark(tp['c1'])} 1. 52주 고가 대비 {tp['from_hi']:+.1f}% (기준 -{HIGH_DOWN_PCT:.0f}% 이내)",
        f"{mark(tp['c2'])} 2. 52주 저가 대비 {tp['from_lo']:+.1f}% (기준 +{LOW_UP_PCT:.0f}% 이상)",
        f"{mark(tp['c3'])} 3. 50일 {tp['s50']:,.0f} > 150일 {tp['s150']:,.0f} > 200일 {tp['s200']:,.0f}",
        f"{mark(tp['c4'])} 4. 종가 {tp['close']:,.0f} 가 150일·200일선 위",
        f"{mark(tp['c5'])} 5. 200일선 상승 중 (약 {tp['rising_m']:.1f}개월째)",
        f"{mark(rs is not None and rs >= RS_MIN)} 7. RS {rs:.0f} (기준 {RS_MIN:.0f} 이상)" if rs is not None else "❔ 7. RS 계산 불가",
    ]
    r = check_vcp(df)
    cons = r.get("cons", [])
    if cons:
        idx = df.index
        lines.append(f"6. 베이스: {idx[r['start']].date()} 고점부터 {r['base_len']}일, 수축 {len(cons)}회")
        for i, k in enumerate(cons, 1):
            tag = " (진행 중)" if k.get("open") else ""
            lines.append(f"   T{i}: {idx[k['hi_i']].date()} {k['hi']:,.0f} → {idx[k['lo_i']].date()} {k['lo']:,.0f}"
                         f"  -{k['depth']:.1f}%{tag}")
        if "vdu" in r:
            lines.append(f"   거래량: 최근10일/50일 {r['vdu']:.0f}% · 마지막 수축/첫 수축 {r['leg_ratio']:.0f}%")
        if "pivot" in r:
            lines.append(f"   피벗 {r['pivot']:,.0f} (현재가 대비 {r['to_pivot']:+.1f}%)")
    st = r.get("status")
    if r.get("ok") and st == "breakout":
        lines.append(f"✅ 6. VCP 돌파 ({r['bo_days']}일 전, 돌파일 거래량 {r['bo_vol']:.0f}%)")
    elif r.get("ok"):
        lines.append("✅ 6. VCP 셋업 (피벗 근접)")
    elif st == "forming":
        lines.append(f"⏳ 6. VCP 형성 중 (모양은 VCP, 남은 과제: {r.get('why', '')})")
    else:
        lines.append(f"❌ 6. VCP 아님: {r.get('why', '')}")
    post(f"🔍 **{t}** 진단 ({df.index[-1].date()} 기준)\n```\n" + "\n".join(lines) + "\n```")


# ───────────── 실행 ─────────────
def main():
    now = dt.datetime.now(NY)
    universes = {g.strip(): load_universe(g.strip()) for g in GROUPS if g.strip()}
    meta = {}
    for g, u in universes.items():
        for row in u.itertuples():
            meta.setdefault(row.ticker, row)
    tickers = list(dict.fromkeys(list(meta) + [t for t, _ in DIAGNOSE]))
    if not meta:
        post("⚠️ VCP 검색기: 종목 목록을 불러오지 못했습니다.")
        return 1

    data, tps = {}, {}
    for t, df in download(tickers):
        if len(df) < 60:
            continue
        data[t] = df
        tp = template(df)
        if tp:
            tps[t] = tp
    if not tps:
        post("⚠️ VCP 검색기: 시세를 받지 못했습니다.")
        return 1
    session = max(df.index[-1].date() for df in data.values())
    if not FORCE and not DIAGNOSE and session != now.date():
        print(f"휴장일로 보임 (최근 거래일 {session}). 종료.")
        return

    # 7. RS 백분위 (스캔 대상 종목 안에서. 진단용 추가 종목은 순위 계산에서 빼고 위치만 매김)
    base = {t: tp["rs_raw"] for t, tp in tps.items() if t in meta and tp["liquid"] and not np.isnan(tp["rs_raw"])}
    ranked = np.sort(np.array(list(base.values())))
    def rs_of(x):
        return float(np.searchsorted(ranked, x, side="right") / len(ranked) * 98 + 1) if len(ranked) and not np.isnan(x) else None

    if DIAGNOSE:
        for t, d in DIAGNOSE:
            if d is None:
                diagnose(t, data.get(t), tps.get(t), rs_of(tps[t]["rs_raw"]) if t in tps else None)
                continue
            # 과거 날짜 기준: 그날까지의 데이터로 템플릿·RS·VCP를 다시 계산
            cut = {k: f[f.index.date <= d] for k, f in data.items() if k in meta or k == t}
            raws = []
            for k, f in cut.items():
                if k in meta and len(f) > 252 and f["Close"].iloc[-1] >= MIN_PRICE:
                    x = rs_raw(f["Close"])
                    if not np.isnan(x):
                        raws.append(x)
            rk = np.sort(np.array(raws))
            f = cut.get(t)
            tp = template(f) if f is not None and len(f) else None
            rs = (float(np.searchsorted(rk, tp["rs_raw"], side="right") / len(rk) * 98 + 1)
                  if tp and len(rk) and not np.isnan(tp["rs_raw"]) else None)
            diagnose(f"{t} @{d}", f if f is not None and len(f) >= 60 else None, tp, rs)
        return

    passed = []
    for t, tp in tps.items():
        if t not in meta or not tp["liquid"] or data[t].index[-1].date() != session:
            continue
        rs = rs_of(tp["rs_raw"])
        if all(tp[f"c{i}"] for i in range(1, 6)) and rs is not None and rs >= RS_MIN:
            passed.append({"t": t, "rs": rs, **tp})

    breakouts, setups, forming, plain = [], [], [], []
    for p in passed:
        r = check_vcp(data[p["t"]])
        if r.get("ok") and r.get("status") == "breakout":
            breakouts.append({**p, **r})
        elif r.get("ok") and r.get("status") == "setup":
            setups.append({**p, **r})
        elif r.get("status") == "forming":
            forming.append({**p, **r})
        else:
            plain.append({**p, "code": r.get("code", "data")})

    # 상태: 돌파 알림 중복 방지, 셋업 신규 표시
    sf = STATE / "vcp_state.json"
    st = json.loads(sf.read_text()) if sf.exists() else {}
    bo_sent = st.get("bo_sent", {})
    prev_setups = set(st.get("setups", [])) if st.get("date") != str(session) else set(st.get("prev_setups", []))
    fresh_bo = []
    for b in breakouts:
        last = bo_sent.get(b["t"])
        if FORCE or not last or (session - dt.date.fromisoformat(last)).days >= REALERT_DAYS:
            fresh_bo.append(b)
            bo_sent[b["t"]] = str(session)
    if not FORCE:
        sf.write_text(json.dumps({"date": str(session), "bo_sent": bo_sent,
                                  "setups": [s["t"] for s in setups], "prev_setups": sorted(prev_setups)}))

    post(f"**📐 미너비니 VCP 검색기 | {session} 마감**\n"
         f"템플릿(1~5·RS {RS_MIN:.0f}+) 통과 **{len(passed)}개** → VCP 베이스 **{len(breakouts) + len(setups)}개** "
         f"(🚀 돌파 {len(breakouts)} · 🎯 피벗 근접 {len(setups)}) · ⏳ 형성 중 {len(forming)}개")

    if fresh_bo:
        fresh_bo.sort(key=lambda r: -r["rs"])
        head = (fit("TICKER", 7) + fit("RS", 4, 1) + fit("고점%", 7, 1) + fit("T", 3, 1) + " " + fit("수축%", 14)
                + fit("피벗", 10, 1) + fit("종가", 10, 1) + fit("돌파%", 7, 1) + fit("VOL%", 6, 1) + fit("D", 3, 1) + "  NAME")
        lines = [f"{r['t']:<7}{r['rs']:>4.0f}{r['from_hi']:>7.1f}{r['T']:>3} {depths_str(r['depths']):<14}"
                 f"{r['pivot']:>10,.0f}{r['close']:>10,.0f}{r['bo_pct']:>+7.1f}{r['bo_vol']:>6.0f}{r['bo_days']:>3}  "
                 f"{'🔊' if r['bo_vol'] >= BREAKOUT_VOL * 100 else '  '}{company_name(r['t'], meta)[:22]}" for r in fresh_bo]
        send_block(f"🚀 VCP 피벗 돌파 ({len(fresh_bo)})", head, lines)

    if setups:
        setups.sort(key=lambda r: r["to_pivot"])
        head = (fit("TICKER", 7) + fit("RS", 4, 1) + fit("고점%", 7, 1) + fit("T", 3, 1) + " " + fit("수축%", 14)
                + fit("피벗", 10, 1) + fit("피벗까지", 9, 1) + fit("VDU%", 6, 1) + fit("베이스", 7, 1) + "  NAME")
        lines = [f"{r['t']:<7}{r['rs']:>4.0f}{r['from_hi']:>7.1f}{r['T']:>3} {depths_str(r['depths']):<14}"
                 f"{r['pivot']:>10,.0f}{r['to_pivot']:>+9.1f}{r['vdu']:>6.0f}{r['base_len']:>6}d  "
                 f"{'🆕' if r['t'] not in prev_setups else '  '}{company_name(r['t'], meta)[:22]}" for r in setups]
        send_block(f"🎯 VCP 셋업 · 피벗 근접 ({len(setups)})", head, lines)

    if forming:
        forming.sort(key=lambda r: r["to_pivot"])
        todo_label = {"last_big": "수축", "vdu": "거래량"}
        head = (fit("TICKER", 7) + fit("RS", 4, 1) + fit("고점%", 7, 1) + " " + fit("수축%", 14)
                + fit("피벗", 10, 1) + fit("피벗까지", 9, 1) + fit("VDU%", 6, 1) + "  " + fit("남은과제", 12) + "NAME")
        lines = [f"{r['t']:<7}{r['rs']:>4.0f}{r['from_hi']:>7.1f} {depths_str(r['depths']):<14}"
                 f"{r['pivot']:>10,.0f}{r['to_pivot']:>+9.1f}{r['vdu']:>6.0f}  "
                 + fit("+".join(todo_label.get(x, x) for x in r.get("todo", [])), 12)
                 + company_name(r["t"], meta)[:22] for r in forming]
        send_block(f"⏳ VCP 형성 중 · 아직 덜 조여짐 ({len(forming)})", head, lines)

    if plain:
        labels = {"short": "베이스 없음 (고점 갱신 중이거나 3주 미만)", "few": "수축 1회 이하 (베이스 초기)",
                  "first_deep": "첫 눌림 40% 초과 (너무 깊은 조정)", "expand": "눌림이 오히려 깊어짐",
                  "lows": "저점이 낮아짐", "last_big": "마지막 눌림이 아직 큼 (15~20% 이상)",
                  "vdu": "거래량이 안 마름", "far": "VCP 모양이지만 피벗까지 8% 초과", "data": "데이터 부족"}
        groups: dict[str, list] = {}
        for p in sorted(plain, key=lambda r: -r["rs"]):
            groups.setdefault(p["code"], []).append(f"{company_name(p['t'], meta) or p['t']}({p['rs']:.0f})")
        order = ["far", "last_big", "vdu", "short", "few", "lows", "expand", "first_deep", "data"]
        lines = [f"· {labels.get(k, k)} [{len(groups[k])}]: " + " ".join(groups[k])
                 for k in order if k in groups]
        buf = "**📋 템플릿은 통과, VCP는 아님 — 이유별 (괄호는 RS)**\n"
        for ln in lines:
            if len(buf) + len(ln) > 1900:
                post(buf.strip())
                buf = ""
            buf += ln[:1900] + "\n"
        post(buf.strip())

    post("수축% = 베이스 안 눌림 폭(오래된→최근) · T = 점점 줄어든 수축 횟수 · 피벗 = 마지막 수축이 시작된 고점\n"
         f"VDU% = 최근 10일 ÷ 50일 평균 거래량 (낮을수록 마름) · VOL% = 돌파일 거래량 ÷ 50일 평균 (🔊 = {BREAKOUT_VOL * 100:.0f}% 이상)\n"
         "남은과제: 수축 = 마지막 눌림이 더 작아져야 함 · 거래량 = 거래량이 더 말라야 함\n"
         f"D = 돌파 후 경과일 · 🆕 = 새로 셋업에 들어온 종목 · RS는 스캔 대상 약 {len(base)}종목 안에서의 순위")
    print(f"완료: 통과 {len(passed)}, 돌파 {len(breakouts)}, 셋업 {len(setups)}")



# ───────────── 한국 시장 어댑터 (미국판 함수 교체) ─────────────
import krcommon as _kr  # noqa: E402
NY = _kr.KST                                   # 시간대: 서울
GROUP_LABEL = _kr.GROUP_LABEL                  # KOSPI / KOSDAQ / 관심종목
load_universe = _kr.load_universe              # 거래대금 상위 + kr_watchlist.txt
download = _kr.make_download("2y", True)  # 야후 .KS / .KQ (SPY → 코스피 지수)
company_name = _kr.company_name                # 한글 종목명


if __name__ == "__main__":
    sys.exit(main())
