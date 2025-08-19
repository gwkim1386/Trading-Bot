#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, logging, asyncio, datetime as dt, re
import numpy as np, pandas as pd
from telegram.ext import Application, MessageHandler, CommandHandler, ContextTypes, filters
from telegram import Update
import yfinance as yf

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)

WINDOWS = [("3M",63), ("6M",126), ("12M",252), ("24M",504), ("28M",588)]
DATE_PATTERNS = ["%Y%m%d", "%Y-%m-%d"]

def ensure_series(obj) -> pd.Series:
    if isinstance(obj, pd.Series): return obj
    if isinstance(obj, pd.DataFrame): return obj.iloc[:, 0]
    arr = np.asarray(obj)
    if arr.ndim == 2 and arr.shape[1] == 1: arr = arr.squeeze(axis=1)
    return pd.Series(arr)

def realized_sigma_tail(returns: pd.Series, tail_n: int, min_samples: int = 20) -> float:
    r = ensure_series(returns).dropna().tail(tail_n)
    if len(r) < max(min_samples, int(tail_n*0.5)): return np.nan
    return float(r.std(ddof=1))

def fetch_close_series_sync(ticker: str, cutoff: dt.date, lookback_days: int = 1400) -> pd.Series:
    start = cutoff - dt.timedelta(days=lookback_days + 60)
    df = yf.download(
        ticker, start=start.isoformat(), end=cutoff.isoformat(),
        auto_adjust=True, progress=False, interval="1d", threads=True
    )
    if df is None or df.empty or "Close" not in df.columns:
        raise ValueError(f"{ticker}: 가격 데이터가 없습니다.")
    s = ensure_series(df["Close"]).dropna()
    return s[s.index.date < cutoff]

async def fetch_close_series(ticker: str, cutoff: dt.date) -> pd.Series:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, fetch_close_series_sync, ticker, cutoff)

def parse_input(text: str) -> tuple[list[str], dt.date]:
    """
    허용 입력:
      - "TQQQ, 20250719"
      - "TQQQ TMF SOXL, 2025-07-19"
      - "TQQQ TMF" (날짜 생략 시 오늘 기준)
    """
    t = text.strip()
    # 쉼표 기준으로 날짜 분리 (한글/공백 다양성 대응)
    parts = [p.strip() for p in re.split(r"\s*,\s*", t, maxsplit=1)]
    tickers_part = parts[0]
    date_part = parts[1] if len(parts) > 1 else None

    # 티커 파싱(공백/쉼표 혼용 허용)
    tickers = [x.strip().upper() for x in re.split(r"[,\s]+", tickers_part) if x.strip()]
    if not tickers:
        raise ValueError("티커를 찾을 수 없습니다. 예) TQQQ, 20250719")

    # 날짜 파싱
    if date_part:
        date_part = date_part.strip()
        last_err = None
        for fmt in DATE_PATTERNS:
            try:
                cutoff = dt.datetime.strptime(date_part, fmt).date()
                break
            except Exception as e:
                last_err = e
                cutoff = None
        if cutoff is None:
            raise ValueError("날짜 형식이 올바르지 않습니다. YYYYMMDD 또는 YYYY-MM-DD 사용.")
    else:
        cutoff = dt.date.today()

    # 미래 날짜 방지(원하면 제거)
    today = dt.date.today()
    if cutoff > today:
        raise ValueError(f"cutoff {cutoff.isoformat()} 는 미래입니다. 오늘({today.isoformat()}) 이하로 입력하세요.")
    return tickers, cutoff

async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "티커와 날짜를 입력해 주세요.\n"
        "예) TQQQ, 20250719   또는   TQQQ TMF SOXL, 2025-07-19\n"
        "날짜 생략 시 오늘 기준으로 계산합니다."
    )

async def on_message(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    # 디버그용 콘솔 출력
    print("받음")

    text = (update.message.text or "").strip()
    if not text:
        await update.message.reply_text("형식: 티커[, 공백] YYYYMMDD  예) TQQQ, 20250719")
        return

    # 입력 파싱 (예: "SOXL, 20250719" 또는 "TQQQ TMF, 2025-07-19" 또는 "SOXL")
    try:
        tickers, cutoff = parse_input(text)
    except ValueError as e:
        await update.message.reply_text(str(e))
        return

    await update.message.reply_text(f"처리 중… (cutoff {cutoff.isoformat()} 미포함)")

    # 결과 메시지 구성 (HTML 사용, 표는 <pre>로 고정폭 렌더링)
    lines = [f"<b>Daily σ Bands</b> (cutoff {cutoff.isoformat()} 미포함)\n"]

    for tkr in tickers:
        try:
            # yfinance 동기 호출을 스레드로: 타임아웃 12초
            s = await asyncio.wait_for(fetch_close_series(tkr, cutoff), timeout=12)
            rets = s.pct_change()
        except asyncio.TimeoutError:
            lines.append(f"<b>{tkr}</b>\n<pre>데이터 응답 지연(타임아웃)</pre>\n")
            continue
        except Exception as e:
            lines.append(f"<b>{tkr}</b>\n<pre>오류: {e}</pre>\n")
            continue

        # 표 헤더 및 행 생성 (고정폭 정렬)
        header = "Window    ±1σ       ±2σ"
        rows = []
        for name, n in WINDOWS:
            sigma = realized_sigma_tail(rets, n)
            if np.isnan(sigma) or sigma <= 0:
                rows.append(f"{name:<7}  N/A       N/A")
            else:
                rows.append(f"{name:<7}  {sigma*100:>6.2f}%   {2*sigma*100:>6.2f}%")

        table_str = "\n".join([header] + rows)
        # HTML에서 코드블록은 ``` 대신 <pre>...</pre> 사용
        lines.append(f"<b>{tkr}</b>\n<pre>{table_str}</pre>\n")

    # 한 번에 전송 (HTML 파서 사용)
    await update.message.reply_html("\n".join(lines).rstrip())


def main():
    token = os.getenv("TELEGRAM_TOKEN")
    if not token:
        raise SystemExit("환경변수 TELEGRAM_TOKEN 이 필요합니다.")
    app = Application.builder().token(token).build()

    async def _post_init(app: Application):
        me = await app.bot.get_me()
        print(f"[봇 기동] id={me.id}, username=@{me.username}")

    app.post_init = _post_init
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))
    app.run_polling(allowed_updates=None)

if __name__ == "__main__":
    main()
