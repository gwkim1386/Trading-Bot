
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Daily & Intraday-Range Volatility Reports (with cutoff date)

- Cutoff(기준일) 이전 데이터만 사용 (cutoff '미포함')
- Report 1: 종가 기준 일간 수익률로 σ 계산 (Daily)
- Report 2: (고가-저가)/전일 종가 = 일중 변동폭으로 σ 계산 (Range)
- 4개 윈도우(1Y=252, 6M=126, 3M=63, 2Y=504) 각각 σ, ±1σ/±2σ 밴드, 신뢰수준/초과확률 표
- 차트는 선택한 윈도우(기본 1Y)의 σ로 정규분포 PDF + 음영
- XlsxWriter가 설치되어 있으면 퍼센트 서식/이미지 삽입까지 적용. 없으면 openpyxl로 폴백.

Usage:
    python daily_range_vol_reports.py --tickers SOXL --cutoff 2025-07-19 --outdir .\out --chart-window 1Y
"""

import argparse
import datetime as dt
from pathlib import Path
import os
import platform
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib import font_manager, rcParams
from scipy.stats import norm

try:
    import yfinance as yf
except Exception:
    raise SystemExit("yfinance가 필요합니다.  pip install yfinance")

WINDOWS_TRADING_DAYS = {"1Y": 252, "6M": 126, "3M": 63, "2Y": 504}

# ----------------------------- Fonts -------------------------------

def setup_korean_font():
    sysname = platform.system()
    candidates = []
    if sysname == "Windows":
        candidates = ["Malgun Gothic", "맑은 고딕"]
    elif sysname == "Darwin":
        candidates = ["AppleGothic"]
    else:
        candidates = ["NanumGothic", "DejaVu Sans"]
    available = [f.name for f in font_manager.fontManager.ttflist]
    for name in candidates:
        if name in available:
            rcParams["font.family"] = name
            rcParams["axes.unicode_minus"] = False
            break

setup_korean_font()

# ----------------------------- Args --------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Daily & Range Vol Reports with cutoff")
    p.add_argument("--tickers", nargs="+", default=["TQQQ", "TMF"], help="예: TQQQ TMF SOXL")
    p.add_argument("--cutoff", default=dt.date.today().isoformat(),
                   help="기준일(YYYY-MM-DD). 이 날짜 '이전' 데이터만 사용")
    p.add_argument("--outdir", default="./out", help="출력 폴더")
    p.add_argument("--lookback-days", type=int, default=1200,
                   help="다운로드 캘린더 일수(2Y+버퍼 권장)")
    p.add_argument("--chart-window", choices=list(WINDOWS_TRADING_DAYS.keys()), default="1Y",
                   help="차트/디테일 표에 사용할 σ 윈도우")
    p.add_argument("--target-moves", default="1,2,3,5",
                   help="관심 변동폭(%), 콤마로 구분. 예: 1,2,3,5")
    p.add_argument("--min-samples", type=int, default=20,
                   help="시리즈 최소 샘플 수(부족하면 자동 제외)")
    return p.parse_args()

# ----------------------------- IO utils ----------------------------

def safe_path(path: Path) -> Path:
    """같은 파일이 열려있거나 존재하면 타임스탬프 붙여서 저장"""
    try:
        if not path.exists():
            return path
    except Exception:
        pass
    ts = dt.datetime.now().strftime("%H%M%S")
    return path.with_name(f"{path.stem}_{ts}{path.suffix}")

def ensure_series(obj) -> pd.Series:
    """여러 타입을 1D pandas Series로 안전 변환"""
    if isinstance(obj, pd.Series):
        return obj
    if isinstance(obj, pd.DataFrame):
        if obj.shape[1] == 1:
            return obj.iloc[:, 0]
        # 여러 컬럼이면 첫 컬럼만 사용 (원하면 명시적으로 에러를 내도 됨)
        return obj.iloc[:, 0]
    import numpy as np
    arr = np.asarray(obj)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr.squeeze(axis=1)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D series-like, got shape {arr.shape}")
    return pd.Series(arr)

# ----------------------------- Data --------------------------------

def download_ohlc(ticker: str, end_date: dt.date, lookback_days: int) -> pd.DataFrame:
    start_date = end_date - dt.timedelta(days=lookback_days + 60)
    df = yf.download(
        ticker, start=start_date.isoformat(), end=end_date.isoformat(),
        auto_adjust=True, progress=False, interval="1d", threads=True
    )
    if df is None or df.empty:
        raise ValueError(f"{ticker}: 데이터가 비어 있습니다.")
    cols = [c for c in ["Open", "High", "Low", "Close"] if c in df.columns]
    if not cols:
        raise ValueError(f"{ticker}: 필요한 컬럼(Open/High/Low/Close)이 없습니다.")
    return df[cols].dropna()

def filter_before_cutoff(df: pd.DataFrame, cutoff: dt.date) -> pd.DataFrame:
    """기준일 이전 데이터만 사용 (cutoff '미포함')"""
    return df[df.index.date < cutoff].copy()

def daily_returns(close: pd.Series) -> pd.Series:
    """전일 종가 대비 오늘 종가 수익률(소수)"""
    return close.pct_change()

def range_vol(high: pd.Series, low: pd.Series, prev_close: pd.Series) -> pd.Series:
    """일중 변동폭 (고-저)/전일 종가, 소수"""
    return (high - low) / prev_close

# --------------------------- Statistics ----------------------------

def realized_sigma_tail(series: pd.Series, tail_n: int) -> float:
    s = pd.Series(series).dropna().tail(tail_n)
    if len(s) < max(20, int(tail_n * 0.5)):
        return np.nan
    return float(s.std(ddof=1))

def sigma_by_windows(series: pd.Series) -> dict:
    return {name: realized_sigma_tail(series, w) for name, w in WINDOWS_TRADING_DAYS.items()}

def band_table_for_sigma(sigma: float, conf_list) -> pd.DataFrame:
    rows = []
    for c in conf_list:
        z = norm.ppf((1 + c) / 2.0)
        if sigma and sigma > 0 and not np.isnan(sigma):
            m = z * sigma
            mv = f"{m*100:.2f}%"
        else:
            mv = None
        rows.append({"두측 신뢰수준": f"{c*100:.2f}%", "z(σ배수)": round(float(z), 4), "±변동폭": mv})
    return pd.DataFrame(rows)

def move_probability_table(sigma: float, target_moves_pct) -> pd.DataFrame:
    rows = []
    for x_pct in target_moves_pct:
        x = x_pct / 100.0
        if not sigma or sigma <= 0 or np.isnan(sigma):
            rows.append({"대상변동(절대)": f"{x_pct:.2f}%", "k(몇 σ)": None,
                         "단측 초과확률 P(R≥x)": None, "양측 초과확률 P(|R|≥x)": None,
                         "양측 신뢰수준 P(|R|≤x)": None})
            continue
        k = x / sigma
        p_one = 1 - norm.cdf(k)
        p_two = 2 * p_one
        rows.append({
            "대상변동(절대)": f"{x_pct:.2f}%",
            "k(몇 σ)": round(float(k), 4),
            "단측 초과확률 P(R≥x)": f"{p_one*100:.3f}%",
            "양측 초과확률 P(|R|≥x)": f"{p_two*100:.3f}%",
            "양측 신뢰수준 P(|R|≤x)": f"{(1-p_two)*100:.3f}%"
        })
    return pd.DataFrame(rows)


# ---------------------------- Excel I/O -----------------------------

def get_excel_writer(dest_path: Path):
    """가능하면 xlsxwriter 사용(서식/이미지), 아니면 openpyxl 폴백"""
    try:
        import xlsxwriter  # noqa: F401
        return pd.ExcelWriter(dest_path, engine="xlsxwriter"), "xlsxwriter"
    except Exception:
        return pd.ExcelWriter(dest_path, engine="openpyxl"), "openpyxl"

def write_report_excel(dest_path: Path,
                       asof: dt.date,
                       cutoff: dt.date,
                       mode_label: str,   # "Daily" or "Range"
                       series_map: dict,  # {ticker: pd.Series (소수)}
                       chart_window: str,
                       target_moves_pct,
                       conf_list=(0.6827, 0.90, 0.95, 0.99, 0.9973),
                       charts_dir: Path | None = None):
    """티커별 표/차트 포함 엑셀 생성. xlsxwriter면 퍼센트 서식/이미지 삽입."""
    charts_dir = charts_dir or dest_path.parent
    dest_path = safe_path(dest_path)

    writer, engine = get_excel_writer(dest_path)
    with writer:
        # 서식 준비(xlsxwriter 전용)
        percent_fmt = pct4_fmt = bold_fmt = None
        if engine == "xlsxwriter":
            wb = writer.book
            percent_fmt = wb.add_format({'num_format': '0.00%'})
            pct4_fmt = wb.add_format({'num_format': '0.0000%'})
            bold_fmt = wb.add_format({'bold': True})

        # Report 시트 생성
        sheet = "Report"
        pd.DataFrame({"_": []}).to_excel(writer, sheet_name=sheet, index=False)
        ws = writer.sheets[sheet]

        # 헤더 작성
        header_lines = [
            f"{mode_label} Volatility Report (기준일 이전 데이터)",
            f"As-of: {asof.isoformat()}   Cutoff: {cutoff.isoformat()} (cutoff 미포함)",
            "Windows: 1Y=252, 6M=126, 3M=63, 2Y=504",
            f"Chart window: {chart_window}   Shaded confidence: 95%"
        ]
        for i, line in enumerate(header_lines):
            if engine == "xlsxwriter" and bold_fmt:
                ws.write(i, 0, line, bold_fmt)
            else:
                ws.cell(row=i+1, column=1, value=line)
        row = 6

        for tkr, ser in series_map.items():
            # 안전 시리즈 (Series 강제, NaN 제거)
            ser = ensure_series(ser).dropna()
            if ser.empty:
                continue

            # 블록 제목
            if engine == "xlsxwriter" and bold_fmt:
                ws.write(row-1, 0, f"[{tkr}] σ (daily) by window", bold_fmt)
            else:
                ws.cell(row=row-1, column=1, value=f"[{tkr}] σ (daily) by window")

            # σ 테이블 헤더
            headers = ["Window", "σ (daily)", "±1σ 변동폭", "±2σ 변동폭"]
            if engine == "xlsxwriter":
                for c, h in enumerate(headers):
                    ws.write(row, c, h, bold_fmt)
            else:
                for c, h in enumerate(headers):
                    ws.cell(row=row, column=c+1, value=h)
            row += 1

            # 값 쓰기
            sigmas = sigma_by_windows(ser)
            for wname in ["1Y", "6M", "3M", "2Y"]:
                s = sigmas.get(wname, np.nan)
                if engine == "xlsxwriter":
                    ws.write(row, 0, wname)
                    if s == s:
                        ws.write_number(row, 1, s, percent_fmt)
                        ws.write_number(row, 2, s, percent_fmt)
                        ws.write_number(row, 3, 2*s, percent_fmt)
                    else:
                        ws.write(row, 1, "N/A")
                        ws.write(row, 2, "N/A")
                        ws.write(row, 3, "N/A")
                else:
                    ws.cell(row=row, column=1, value=wname)
                    ws.cell(row=row, column=2, value=s if s == s else None)
                    ws.cell(row=row, column=3, value=s if s == s else None)
                    ws.cell(row=row, column=4, value=(2*s) if s == s else None)
                row += 1

            row += 1

            # 상세 표/차트는 chart_window σ 사용
            sigma_for_tables = sigmas.get(chart_window, np.nan)

            bt = band_table_for_sigma(sigma_for_tables, conf_list)
            bt.to_excel(writer, sheet_name=sheet, startrow=row-1, startcol=0, index=False)
            row += len(bt) + 2

            mt = move_probability_table(sigma_for_tables, target_moves_pct)
            mt.to_excel(writer, sheet_name=sheet, startrow=row-1, startcol=0, index=False)
            row += len(mt) + 1

            # 차트 파일 만들고, xlsxwriter면 이미지 삽입 / 아니면 경로만 기록
            chart_path = plot_normal_pdf(charts_dir, tkr, sigma_for_tables, 95.0, f"{mode_label}_{chart_window}")
            if engine == "xlsxwriter" and chart_path and Path(chart_path).exists():
                ws.insert_image(row-1, 0, str(chart_path), {'x_scale': 0.9, 'y_scale': 0.9})
                row += 22
            else:
                # 경로 텍스트 기록
                if engine == "xlsxwriter":
                    ws.write(row-1, 0, f"Chart saved: {chart_path}")
                else:
                    ws.cell(row=row, column=1, value=f"Chart saved: {chart_path}")
                row += 3

        # Raw(%) 시트: 여러 시리즈 안전 결합 (index 다른 경우도 OK)
        series_clean = {}
        for t, s in series_map.items():
            s = ensure_series(s).dropna()
            if len(s) >= 2:
                series_clean[t] = s

        if series_clean:
            raw = pd.concat(series_clean, axis=1)
            raw.index.name = "Date"
            raw_pct = raw.tail(120) * 100.0
            raw_pct.to_excel(writer, sheet_name="Raw(%)")
        else:
            pd.DataFrame({"note": ["No sufficient raw series to display (check cutoff / history length)"]}) \
              .to_excel(writer, sheet_name="Raw(%)", index=False)

    print(f"✓ Saved: {dest_path}")





# ----------------------------- Main ---------------------------------

def main():
    args = parse_args()
    tickers = args.tickers
    cutoff = dt.date.fromisoformat(args.cutoff)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    asof = dt.date.today()
    target_moves_pct = [float(x) for x in args.target_moves.split(",") if x.strip()]
    min_samples = int(args.min_samples)

    # 데이터 다운로드 & 컷오프 적용
    ohlc_map = {}
    for t in tickers:
        df = download_ohlc(t, end_date=cutoff, lookback_days=args.lookback_days)
        df = filter_before_cutoff(df, cutoff)
        if df.empty:
            raise SystemExit(f"{t}: 기준일 이전 데이터가 없습니다. cutoff/기간을 조정하세요.")
        ohlc_map[t] = df

    # 시리즈 생성 (소수)
    daily_series = {}
    range_series = {}
    for t, df in ohlc_map.items():
        close = df["Close"]
        dr = daily_returns(close).dropna()
        if len(dr) >= min_samples:
            daily_series[t] = dr

        prev_close = close.shift(1)
        rng = range_vol(df["High"], df["Low"], prev_close).dropna()
        if len(rng) >= min_samples:
            range_series[t] = rng

    if not daily_series and not range_series:
        raise SystemExit("기준일 이전 데이터가 너무 짧아 통계를 만들 수 없습니다. cutoff/기간을 늘려보세요.")

    # Report 1: Daily returns
    daily_path = outdir / f"US_Vol_Report_{cutoff.isoformat()}_Daily.xlsx"
    write_report_excel(
        dest_path=daily_path,
        asof=asof,
        cutoff=cutoff,
        mode_label="Daily",
        series_map=daily_series,
        chart_window=args.chart_window,
        target_moves_pct=target_moves_pct,
        charts_dir=outdir
    )

    # Report 2: Intraday range
    range_path = outdir / f"US_Vol_Report_{cutoff.isoformat()}_Range.xlsx"
    write_report_excel(
        dest_path=range_path,
        asof=asof,
        cutoff=cutoff,
        mode_label="Range",
        series_map=range_series,
        chart_window=args.chart_window,
        target_moves_pct=target_moves_pct,
        charts_dir=outdir
    )

if __name__ == "__main__":
    main()
