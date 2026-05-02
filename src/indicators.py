import numpy as np

from config.config import MA_SHORT, MA_LONG


def compute_rsi(series, window=14):
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    return rsi


def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()

    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_histogram = macd - macd_signal

    return macd, macd_signal, macd_histogram


def add_indicators(df):
    df = df.copy()

    price_col = "adj_close" if "adj_close" in df.columns else "close"

    df["daily_return"] = df[price_col].pct_change()
    df["cumulative_return"] = (1 + df["daily_return"].fillna(0)).cumprod()

    df["ma50"] = df[price_col].rolling(window=MA_SHORT).mean()
    df["ma200"] = df[price_col].rolling(window=MA_LONG).mean()

    df["rsi_14"] = compute_rsi(df[price_col], window=14)

    macd, macd_signal, macd_histogram = compute_macd(df[price_col])
    df["macd"] = macd
    df["macd_signal"] = macd_signal
    df["macd_histogram"] = macd_histogram

    df["volatility_30"] = df["daily_return"].rolling(30).std() * np.sqrt(252)

    return df