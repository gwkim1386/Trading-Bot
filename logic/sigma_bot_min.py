#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sigma Bot (Minimum): 티커를 입력하면 Daily σ와 ±1σ, ±2σ만 반환
- Daily Close-to-Close 기준
- cutoff 이전 데이터만 사용 (미포함)
- 윈도우: 1Y=252, 6M=126, 3M=63, 2Y=504
- 텔레그램 명령: /sigma <티커> [YYYY-MM-DD] [1Y|6M|3M|2Y]
  예) /sigma AAPL 2025-08-18 1Y

필요 패키지:
  python-telegram-bot==21.4
  yfinance pandas numpy pytz (또는 zoneinfo) matplotlib(불필요), scipy(불필요)
"""

import os
import math
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

# ----------------------------- Settings -----------------------------
KST = ZoneInfo("Asia/Seoul")
WINDOWS = {"1Y": 252, "6M": 126, "3M": 63, "2Y": 504}
DEFAULT_WINDOW = "1Y"

# ----------------------------- Utils --------------------------------
def parse_date_yyyy_mm_dd(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()

def cutoff_filter(df: pd.DataFrame, cutoff: date, include: bool = False) -> pd.DataFrame:
    # include=False → cutoff '미포함'
    return df[df.index.date <= cutoff] if include else df[df.index.date < cutoff]

def fetch_ohlc(ticker: str, cutoff: date, lookback_days: int = 1200) -> pd.DataFrame:
    start = cutoff - timedelta(days=lookback_days + 60)
    df = yf.download(
        ticker,
        start=start.isoformat(),
        end=(cutoff + timedelta(days=1)).isoformat(),
        auto_adjust=True, progress=False, interval="1d", threads=True
    )
    if df is None or df.empty or "Close" not in df.columns:
        raise RuntimeError(f"{ticker}: 데이터 없음/형식 오류")
    return df[["Close"]].dropna()

def daily_returns(close: pd.Series) -> pd.Series:
    return close.pct_change()

def realized_sigma_tail(series: pd.Series, tail_n: int) -> float:
    s = pd.Series(series).dropna().tail(tail_n)
    if len(s) < max(20, int(tail_n*0.5)):
        return float("nan")
    return float(s.std(ddof=1))

# ----------------------------- Core ---------------------------------
def calc_sigmas_for_window(ticker: str, cutoff: date, window_key: str):
    if window_key not in WINDOWS:
        raise ValueError("윈도우는 1Y/6M/3M/2Y 중 하나여야 합니다.")
    n = WINDOWS[window_key]

    df = fetch_ohlc(ticker, cutoff)
    df = cutoff_filter(df, cutoff, include=False)
    if df.empty:
        raise RuntimeError(f"{ticker}: 기준일 이전 데이터 없음")

    dr = daily_returns(df["Close"]).dropna()
    s = realized_sigma_tail(dr, n)  # daily σ

    return s  # daily σ (decimal, e.g., 0.0123 = 1.23%)

def format_sigma_message(ticker: str, cutoff: date, window_key: str) -> str:
    s = calc_sigmas_for_window(ticker, cutoff, window_key)
    if not s or math.isnan(s) or s <= 0:
        return (f"[{ticker}] {window_key} 기준\n"
                f"- 샘플 부족 또는 σ 계산 불가")
    one = s*100
    two = 2*s*100
    return (f"[{ticker}] Daily σ ({window_key})\n"
            f"- σ ≈ {one:.2f}%\n"
            f"- ±1σ ≈ {one:.2f}%  |  ±2σ ≈ {two:.2f}%\n"
            f"(Cutoff<{cutoff.isoformat()}, 종가-종가 기준)")

# --------------------------- Telegram Bot ---------------------------
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "안녕하세요! σ(표준편차) 봇입니다.\n"
        "사용법: /sigma <티커> [YYYY-MM-DD] [1Y|6M|3M|2Y]"
    )

async def cmd_sigma(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        return await update.message.reply_text(
            "사용법: /sigma <티커> [YYYY-MM-DD] [1Y|6M|3M|2Y]\n예) /sigma AAPL 2025-08-18 1Y"
        )
    ticker = context.args[0].upper()
    cutoff = datetime.now(KST).date()
    window = DEFAULT_WINDOW

    if len(context.args) >= 2:
        # 두 번째 인자가 날짜인지 윈도우인지 판별
        a2 = context.args[1]
        try:
            cutoff = parse_date_yyyy_mm_dd(a2)
        except Exception:
            if a2.upper() in WINDOWS:
                window = a2.upper()
            else:
                return await update.message.reply_text("날짜 형식은 YYYY-MM-DD, 윈도우는 1Y/6M/3M/2Y")
    if len(context.args) >= 3:
        w = context.args[2].upper()
        if w in WINDOWS:
            window = w

    try:
        msg = format_sigma_message(ticker, cutoff, window)
        await update.message.reply_text(msg)
    except Exception as e:
        await update.message.reply_text(f"에러: {e}")

def build_app(token: str):
    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("sigma", cmd_sigma))
    return app

# ------------------------------ Main --------------------------------
if __name__ == "__main__":
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if token:
        app = build_app(token)
        app.run_polling(allowed_updates=Update.ALL_TYPES)
    else:
        # 토큰이 없으면 CLI 모드로 간단 테스트
        import sys
        if len(sys.argv) < 2:
            print("사용법(CLI): python sigma_bot_min.py <티커> [YYYY-MM-DD] [1Y|6M|3M|2Y]")
            print("예: python sigma_bot_min.py AAPL 2025-08-18 1Y")
            raise SystemExit(1)
        ticker = sys.argv[1].upper()
        cutoff = datetime.now(KST).date()
        window = DEFAULT_WINDOW
        if len(sys.argv) >= 3:
            a2 = sys.argv[2]
            try:
                cutoff = parse_date_yyyy_mm_dd(a2)
            except Exception:
                if a2.upper() in WINDOWS: window = a2.upper()
        if len(sys.argv) >= 4:
            w = sys.argv[3].upper()
            if w in WINDOWS: window = w
        print(format_sigma_message(ticker, cutoff, window))
