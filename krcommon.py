"""
한국 주식 스캐너 공통 모듈
- 종목 목록: KRX 일별 시세 (FinanceDataReader가 GitHub에 올려 두는 KRX 캐시)
  → 시가총액 MIN_MARCAP(기본 1조 원) 이상 + 최근 20거래일 평균 거래대금 MIN_TURNOVER(기본 100억 원) 이상인 코스피·코스닥 보통주
  (우선주·스팩·관리종목·투자주의환기종목 제외) + kr_watchlist.txt
- 시세: 야후 파이낸스 (코스피 = 코드.KS, 코스닥 = 코드.KQ)
- 미국판 스캐너 코드는 그대로 두고, 종목 목록·시세·종목명 함수만 이 모듈 것으로 바꿔 끼웁니다.
"""
import datetime as dt
import io
import json
import os
import re
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

ROOT = Path(__file__).parent
COMMON = ROOT / "state" / "common"
COMMON.mkdir(parents=True, exist_ok=True)
KST = ZoneInfo("Asia/Seoul")
MIN_TURNOVER = float(os.environ.get("MIN_TURNOVER", "10000000000"))  # 20일 평균 거래대금(원) — 100억
MIN_MARCAP = float(os.environ.get("MIN_MARCAP", "1000000000000"))     # 시가총액(원) — 1조
TURNOVER_DAYS = int(os.environ.get("TURNOVER_DAYS", "20"))
CACHE_URL = "https://raw.githubusercontent.com/FinanceData/fdr_krx_data_cache/refs/heads/master/data/listing/krx/{d}.csv"
GROUP_LABEL = {"kospi": "KOSPI", "kosdaq": "KOSDAQ", "watchlist": "관심종목"}
ALIAS = {"SPY": "^KS11", "KOSPI": "^KS11", "KOSDAQ": "^KQ11"}           # 미국판 코드가 SPY를 부르면 코스피 지수로
UA = {"User-Agent": "Mozilla/5.0 (kr-scanners)"}


# ───────────── 종목 목록 ─────────────
def _fetch_day(d: dt.date) -> pd.DataFrame | None:
    try:
        r = requests.get(CACHE_URL.format(d=d.isoformat()), headers=UA, timeout=30)
        if r.status_code != 200:
            return None
        df = pd.read_csv(io.StringIO(r.content.decode("utf-8-sig")), index_col=0,
                         dtype={"Code": str, "Dept": str, "MarketId": str})
        for c in ("Amount", "Marcap", "Close"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        # 장 시작 전에는 그날 파일이 있어도 시세가 비어 있음 → 그 날은 건너뜀
        if len(df) < 1000 or df["Amount"].fillna(0).sum() <= 0 or (df["Marcap"].fillna(0) > 0).sum() < 1000:
            return None
        return df
    except Exception as e:
        print(f"[경고] KRX 목록 {d} 실패: {e}")
        return None


def _is_common(row) -> bool:
    name, dept = str(row["Name"]), str(row.get("Dept", "") or "")
    if row["Market"] not in ("KOSPI", "KOSDAQ", "KOSDAQ GLOBAL"):
        return False
    if "스팩" in name or "SPAC" in dept or "관리종목" in dept or "투자주의환기" in dept:
        return False
    if re.search(r"(우|우B|우C|\(전환\))$", name):                 # 우선주
        return False
    return True


def build_universe() -> pd.DataFrame:
    """최근 TURNOVER_DAYS 거래일 KRX 시세로 평균 거래대금 계산 → 기준 이상 종목. state/common/universe.csv 저장"""
    today = dt.datetime.now(KST).date()
    days, d = [], today
    for _ in range(45):                                               # 달력일 45일 안에서 거래일 20개
        if d.weekday() < 5:
            df = _fetch_day(d)
            if df is not None:
                days.append(df[["Code", "Name", "Market", "Dept", "Amount", "Close", "Marcap"]].assign(date=d))
                if len(days) >= TURNOVER_DAYS:
                    break
        d -= dt.timedelta(days=1)
    if not days:
        raise RuntimeError("KRX 종목 목록을 받지 못했습니다")
    latest = days[0]
    latest = latest[latest.apply(_is_common, axis=1)]
    amt = pd.concat(days).groupby("Code")["Amount"].mean()
    latest = latest.assign(turnover=latest["Code"].map(amt))
    uni = latest[(latest["turnover"] >= MIN_TURNOVER) & (latest["Marcap"] >= MIN_MARCAP)]
    out = pd.DataFrame({
        "ticker": uni["Code"].astype(str),
        "name": uni["Name"].astype(str),
        "market": uni["Market"].map(lambda m: "KOSPI" if m == "KOSPI" else "KOSDAQ"),
        "turnover": uni["turnover"].round(0),
        "marcap": uni["Marcap"],
    }).sort_values("turnover", ascending=False)
    out.to_csv(COMMON / "universe.csv", index=False)
    # 전체 이름표 (관심종목·진단용 종목명 조회)
    names = latest[["Code", "Name", "Market"]].rename(columns={"Code": "ticker", "Name": "name", "Market": "market"})
    names["market"] = names["market"].map(lambda m: "KOSPI" if m == "KOSPI" else "KOSDAQ")
    names.to_csv(COMMON / "names.csv", index=False)
    (COMMON / "universe_meta.json").write_text(json.dumps(
        {"date": str(days[0]["date"].iloc[0]), "days": len(days), "count": len(out), "built": str(today)}))
    print(f"한국 종목 목록: 기준일 {days[0]['date'].iloc[0]}, 시가총액 {MIN_MARCAP / 1e12:g}조 이상 + "
          f"{len(days)}거래일 평균 거래대금 {MIN_TURNOVER / 1e8:.0f}억 이상 {len(out)}종목")
    return out


def _universe() -> pd.DataFrame:
    p = COMMON / "universe.csv"
    meta = COMMON / "universe_meta.json"
    fresh = False
    if p.exists() and meta.exists():
        built = json.loads(meta.read_text()).get("built", "")
        fresh = built == str(dt.datetime.now(KST).date()) or os.environ.get("UNIVERSE_REUSE") == "1"
    if not fresh:
        try:
            return build_universe()
        except Exception as e:
            print(f"[경고] 종목 목록 갱신 실패, 지난 목록 사용: {e}")
    if not p.exists():
        return pd.DataFrame(columns=["ticker", "name", "market", "turnover"])
    return pd.read_csv(p, dtype={"ticker": str})


def _names() -> dict[str, tuple[str, str]]:
    p = COMMON / "names.csv"
    if not p.exists():
        _universe()
    if not p.exists():
        return {}
    df = pd.read_csv(p, dtype={"ticker": str})
    return {r.ticker: (r.name, r.market) for r in df.itertuples()}


def _watchlist() -> pd.DataFrame:
    p = ROOT / "kr_watchlist.txt"
    names = _names()
    by_name = {v[0]: k for k, v in names.items()}
    rows = []
    if p.exists():
        for ln in p.read_text(encoding="utf-8").splitlines():
            ln = ln.split("#")[0].strip()
            if not ln:
                continue
            key = ln.partition(",")[0].strip()
            code = key if key in names else by_name.get(key, key)        # 코드 또는 종목명
            nm, mkt = names.get(code, (key, "KOSPI"))
            rows.append({"ticker": code, "name": nm, "industry": "관심종목", "market": mkt})
    return pd.DataFrame(rows, columns=["ticker", "name", "industry", "market"]).drop_duplicates("ticker")


def load_universe(group: str) -> pd.DataFrame:
    """미국판의 load_universe 대체. group = kospi / kosdaq / watchlist"""
    if group == "watchlist":
        return _watchlist()
    u = _universe()
    u = u[u["market"] == GROUP_LABEL.get(group, group.upper())]
    return pd.DataFrame({"ticker": u["ticker"], "name": u["name"], "industry": u["market"]})


# ───────────── 시세 ─────────────
def yahoo_symbol(code: str) -> str:
    if code in ALIAS:
        return ALIAS[code]
    if code.startswith("^") or "." in code:
        return code
    names = _names()
    if code not in names:                                             # 종목명으로 넣은 경우
        by_name = {v[0]: k for k, v in names.items()}
        code = by_name.get(code, code)
    mkt = names.get(code, ("", "KOSPI"))[1]
    return f"{code}.{'KS' if mkt == 'KOSPI' else 'KQ'}"


def make_download(period: str = "2y", auto_adjust: bool = True, chunk_size: int = 200):
    """미국판 download(tickers) 대체. 6자리 코드(또는 종목명)를 받아 (원래 키, 일봉)을 차례로 돌려줌"""
    def download(tickers: list[str]):
        import yfinance as yf
        sym = {t: yahoo_symbol(t) for t in tickers}
        keys = list(sym)
        for i in range(0, len(keys), chunk_size):
            chunk = keys[i:i + chunk_size]
            syms = [sym[k] for k in chunk]
            try:
                data = yf.download(syms, period=period, interval="1d", group_by="ticker",
                                   auto_adjust=auto_adjust, threads=True, progress=False)
            except Exception as e:
                print(f"[경고] 시세 다운로드 실패({chunk[0]}…): {e}")
                continue
            if data is None or data.empty:
                continue
            if not isinstance(data.columns, pd.MultiIndex):
                data = pd.concat({syms[0]: data}, axis=1)
            have = set(data.columns.get_level_values(0))
            for k in chunk:
                s = sym[k]
                if s in have:
                    df = data[s].dropna(subset=["Close"])
                    df = df[df["Volume"].fillna(0) >= 0]
                    if len(df):
                        yield k, df
            time.sleep(1)
    return download


def download_bars(tickers: list[str], interval: str, period: str) -> dict[str, pd.DataFrame]:
    """미국판 MA 스캐너 download_bars 대체 (주봉·월봉)"""
    import yfinance as yf
    out = {}
    sym = {t: yahoo_symbol(t) for t in tickers}
    keys = list(sym)
    for i in range(0, len(keys), 100):
        chunk = keys[i:i + 100]
        syms = [sym[k] for k in chunk]
        try:
            data = yf.download(syms, period=period, interval=interval, group_by="ticker",
                               auto_adjust=False, threads=True, progress=False)
        except Exception as e:
            print(f"[경고] {interval} 시세 다운로드 실패: {e}")
            continue
        if data is None or data.empty:
            continue
        if not isinstance(data.columns, pd.MultiIndex):
            data = pd.concat({syms[0]: data}, axis=1)
        have = set(data.columns.get_level_values(0))
        for k in chunk:
            if sym[k] in have:
                out[k] = data[sym[k]].dropna(subset=["Close", "Volume"])
        time.sleep(1)
    return out


def company_name(ticker: str, *_args) -> str:
    return _names().get(ticker, ("", ""))[0]


def market_cap(ticker: str, price: float) -> None:
    return None
