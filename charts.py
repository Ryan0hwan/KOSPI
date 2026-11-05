"""
디스코드용 차트 (PNG) 공통 모듈 — 미장·국장 시스템이 같이 씀
- 어두운 테마 (디스코드 배경에 맞춤), 한글 글꼴 자동 설정 (나눔고딕이 설치돼 있으면 사용)
- send(fig, content): 웹후크로 그림 첨부. 웹후크가 없으면 charts_out/ 폴더에 저장 (테스트용)
차트 오류가 나도 기존 표 알림에는 영향이 없도록, 호출하는 쪽에서 safe()로 감쌉니다.
"""
import io
import json
import os
import time
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

BG, PANEL, GRID, TEXT, SUB = "#0e0f12", "#15171c", "#2a2d35", "#e6e6e6", "#9aa0a6"
GREEN, RED, BLUE, PURPLE, GRAY, YELLOW = "#3fb950", "#f85149", "#4c8dff", "#b083f0", "#6e7681", "#d29922"


def _font():
    for name in ("NanumGothic", "NanumBarunGothic", "Noto Sans CJK KR", "Noto Sans KR", "Malgun Gothic"):
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            return name
    for p in ("/usr/share/fonts/truetype/nanum/NanumGothic.ttf",):
        if os.path.exists(p):
            font_manager.fontManager.addfont(p)
            return font_manager.FontProperties(fname=p).get_name()
    return "DejaVu Sans"


plt.rcParams.update({
    "font.family": _font(), "axes.unicode_minus": False,
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "axes.edgecolor": GRID, "axes.labelcolor": SUB, "text.color": TEXT,
    "xtick.color": SUB, "ytick.color": TEXT, "grid.color": GRID, "grid.linestyle": ":",
    "font.size": 11,
})


def safe(fn, *a, **k):
    """차트 하나가 실패해도 시스템은 계속"""
    try:
        return fn(*a, **k)
    except Exception as e:
        print(f"[경고] 차트 실패: {e}")
        traceback.print_exc()


def send(fig, content: str = "", filename: str = "chart.png", webhook: str | None = None):
    webhook = os.environ.get("DISCORD_WEBHOOK_URL", "") if webhook is None else webhook
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    if not webhook:
        out = Path("charts_out")
        out.mkdir(exist_ok=True)
        (out / filename).write_bytes(buf.getvalue())
        print(f"[차트 저장] {out / filename} {content}")
        return
    import requests
    for _ in range(5):
        r = requests.post(webhook, data={"payload_json": json.dumps({"content": content[:1900]})},
                          files={"file": (filename, buf.getvalue(), "image/png")}, timeout=60)
        if r.status_code == 429:
            time.sleep(float(r.json().get("retry_after", 2)) + 0.5)
            continue
        r.raise_for_status()
        time.sleep(0.6)
        return


def _style(ax, title: str, subtitle: str = ""):
    ax.set_title(title, loc="left", fontsize=14, fontweight="bold", color=TEXT, pad=22 if subtitle else 10)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=10, color=SUB, va="bottom")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x")
    ax.set_axisbelow(True)


def hbar(items: list[tuple[str, float]], title: str, subtitle: str = "", unit: str = "%",
         ref: tuple[str, float] | None = None, fmt: str = "{:+.1f}"):
    """가로 막대 (위 = 큰 값). items = [(라벨, 값)] 또는 [(라벨, 값, 색)]"""
    items = sorted(items, key=lambda x: x[1])
    labels, vals = [x[0] for x in items], [x[1] for x in items]
    colors = [x[2] if len(x) > 2 else (GREEN if x[1] >= 0 else RED) for x in items]
    h = max(3.2, 0.36 * len(items) + 1.4)
    fig, ax = plt.subplots(figsize=(10, h))
    ax.barh(range(len(vals)), vals, color=colors, height=0.68)
    ax.set_yticks(range(len(vals)), labels)
    ax.axvline(0, color=SUB, lw=0.8)
    span = (max(vals + [0]) - min(vals + [0])) or 1
    for i, v in enumerate(vals):
        ax.text(v + (span * 0.01 if v >= 0 else -span * 0.01), i, fmt.format(v) + unit,
                va="center", ha="left" if v >= 0 else "right", fontsize=9, color=TEXT)
    ax.set_xlim(min(vals + [0]) - span * 0.12, max(vals + [0]) + span * 0.12)
    if ref:
        ax.axvline(ref[1], color=YELLOW, lw=1.2, ls="--")
        ax.text(ref[1], len(vals) - 0.4, f" {ref[0]} {fmt.format(ref[1])}{unit}", color=YELLOW, fontsize=9, va="bottom")
    _style(ax, title, subtitle)
    return fig


def hbar_pair(labels: list[str], a: list[float], b: list[float], names=("외국인", "기관"), title: str = "",
              subtitle: str = "", unit: str = "억", colors=(BLUE, PURPLE), sep_after: list[int] | None = None):
    """종목·항목별 막대 두 개(예: 외국인/기관). 위에서부터 labels 순서대로"""
    n = len(labels)
    h = max(3.5, 0.5 * n + 1.6)
    fig, ax = plt.subplots(figsize=(10, h))
    y = list(range(n))[::-1]
    ax.barh([v + 0.19 for v in y], a, height=0.36, color=colors[0], label=names[0])
    ax.barh([v - 0.19 for v in y], b, height=0.36, color=colors[1], label=names[1])
    ax.set_yticks(y, labels)
    ax.axvline(0, color=SUB, lw=0.8)
    allv = [x for x in a + b if x == x] + [0]
    span = (max(allv) - min(allv)) or 1
    for vals, off in ((a, 0.19), (b, -0.19)):
        for yi, v in zip(y, vals):
            if v == v:
                ax.text(v + (span * 0.01 if v >= 0 else -span * 0.01), yi + off, f"{v:+,.0f}",
                        va="center", ha="left" if v >= 0 else "right", fontsize=8, color=TEXT)
    ax.set_xlim(min(allv) - span * 0.15, max(allv) + span * 0.15)
    for k in sep_after or []:
        ax.axhline(n - 1 - k - 0.5, color=GRID, lw=1)
    ax.legend(loc="lower right", bbox_to_anchor=(1, 1.0), ncol=2, frameon=False, labelcolor=TEXT, fontsize=9)
    _style(ax, title, subtitle)
    ax.set_xlabel(unit)
    return fig


def quadrant(points: list[tuple[str, float, float]], title: str, subtitle: str = "",
             xlabel: str = "", ylabel: str = "", label_if=None, mid: float = 50):
    """4분면 산점도. points = [(라벨, x, y)], x·y 0~100 백분위"""
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.axvline(mid, color=GRID, lw=1)
    ax.axhline(mid, color=GRID, lw=1)
    quad = [((mid + 100) / 2, (mid + 100) / 2, "주도 (1주·1개월 모두 강함)", GREEN),
            (mid / 2, (mid + 100) / 2, "막 강해짐 (1주 강, 1개월 약)", BLUE),
            ((mid + 100) / 2, mid / 2, "약해지는 중 (1주 약, 1개월 강)", YELLOW),
            (mid / 2, mid / 2, "약세", RED)]
    for qx, qy, txt, col in quad:
        ax.text(qx, qy, txt, ha="center", va="center", fontsize=13, color=col, alpha=0.18, fontweight="bold")
    for lab, x, y in points:
        col = GREEN if x >= mid and y >= mid else BLUE if y >= mid else YELLOW if x >= mid else RED
        ax.scatter(x, y, s=28, color=col, alpha=0.85, edgecolors="none")
        if label_if is None or label_if(lab, x, y):
            ax.annotate(lab, (x, y), xytext=(4, 3), textcoords="offset points", fontsize=8, color=TEXT)
    ax.set_xlim(-3, 103)
    ax.set_ylim(-3, 103)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel, color=SUB)
    ax.tick_params(axis="y", colors=SUB)
    _style(ax, title, subtitle)
    ax.grid(True)
    ax.spines["left"].set_visible(True)
    return fig
