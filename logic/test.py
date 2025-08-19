#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Console bands only:
- Cutoff(기준일) '이전' 데이터만 사용(미포함)
- 각 창(3M, 6M, 12M, 28M, 24M)에 대해 일간 수익률 σ 계산 후 ±1σ/±2σ(%)만 출력
- 데이터: Yahoo Finance (Adjusted Close)
Usage:
    python vol_bands_console.py --tickers TQQQ TMF --cutoff 2025-07-19
"""

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception:
    raise SystemExit("yfinance가 필요합니다.  pip install yfinance")

# 출력 순서 유지!
WINDOWS = [
    ("3M",  63),
    ("6M",  126),
    ("12M", 252),
    ("28M", 588),
    ("24M", 504),
]

def parse_args():
    p = argparse.ArgumentParser(description="Print ±1σ/±2σ bands only (3M,6M,12M,28M,24M)")
    p.add_argument("--tickers", nargs="+", required=True, help="예: TQQQ TMF SOXL")
    p.add_argument("--cutoff", required=True, help="기준일 YYYY-MM-DD (해당일 미포함)")
    p.add_argument("--lookback-days", type=int, default=1400,
                   help="다운로드 캘린더 일수(최대 창 28M 커버 + 버퍼)")
    p.add_argument("--min-samples", type=int, default=20, help="최소 표본 수(부족 시 NaN)")
    return p.parse_args()

def ensure_series(obj) -> pd.Series:
    if isinstance(obj, pd.Series):
        return obj
    if isinstance(obj, pd.DataFrame):
        if obj.shape[1] == 1:
            return obj.iloc[:, 0]
        return obj.iloc[:, 0]
    import numpy as np
    arr = np.asarray(obj)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr.squeeze(axis=1)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D, got shape {arr.shape}")
    return pd.Series(arr)

def fetch_close_series(ticker: str, cutoff: dt.date, lookback_days: int) -> pd.Series:
    start = cutoff - dt.timedelta(days=lookback_days + 60)
    df = yf.download(
        ticker,
        start=start.isoformat(),
        end=cutoff.isoformat(),            # cutoff '미포함'
        auto_adjust=True,
        progress=False,
        interval="1d",
        threads=True
    )
    if df is None or df.empty or "Close" not in df.columns:
        raise ValueError(f"{ticker}: 가격 데이터가 없습니다.")
    s = ensure_series(df["Close"]).dropna()
    # 혹시 모를 미래/동일일 데이터 제거(미포함 보장)
    s = s[s.index.date < cutoff]
    return s

def realized_sigma_tail(returns: pd.Series, tail_n: int, min_samples: int) -> float:
    r = ensure_series(returns).dropna().tail(tail_n)
    if len(r) < max(min_samples, int(tail_n*0.5)):
        return np.nan
    return float(r.std(ddof=1))

def main():
    args = parse_args()
    tickers = [t.upper() for t in args.tickers]
    cutoff = dt.date.fromisoformat(args.cutoff)

    print(f"As-of: {dt.date.today().isoformat()} | Cutoff: {cutoff.isoformat()} (미포함)")
    print("Windows (trading days): 3M=63, 6M=126, 12M=252, 28M=588, 24M=504\n")

    for tkr in tickers:
        try:
            close = fetch_close_series(tkr, cutoff, args.lookback_days)
            rets = close.pct_change()
        except Exception as e:
            print(f"[{tkr}] 오류: {e}\n")
            continue

        print(f"[{tkr}] ±1σ / ±2σ bands (daily returns)")
        print("Window  ±1σ       ±2σ")
        for name, n in WINDOWS:
            sigma = realized_sigma_tail(rets, n, args.min_samples)
            if np.isnan(sigma) or sigma <= 0:
                p1 = p2 = "N/A"
            else:
                p1 = f"{sigma*100:.2f}%"
                p2 = f"{2*sigma*100:.2f}%"
            print(f"{name:<6} {p1:<9} {p2:<9}")
        print()  # blank line between tickers

if __name__ == "__main__":
    main()
