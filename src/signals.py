from config.config import (
    SIGNAL_RSI_THRESHOLD,
    SIGNAL_USE_SHORT,
    USE_EMA,
    EMA_SHORT,
    EMA_LONG,
    VOL_FILTER_ENABLED,
    VOL_LOOKBACK,
    VOL_THRESHOLD
)


def generate_signals(df):
    df = df.copy()

    # =========================
    # Trend Signal
    # =========================
    if USE_EMA:
        df["ema_short"] = df["close"].ewm(span=EMA_SHORT, adjust=False).mean()
        df["ema_long"] = df["close"].ewm(span=EMA_LONG, adjust=False).mean()

        trend_up = df["ema_short"] > df["ema_long"]
        trend_down = df["ema_short"] < df["ema_long"]
    else:
        trend_up = df["ma50"] > df["ma200"]
        trend_down = df["ma50"] < df["ma200"]

    # =========================
    # Volatility Filter (FIXED)
    # =========================
    if VOL_FILTER_ENABLED:
        df["vol_filter"] = df["daily_return"].rolling(VOL_LOOKBACK).std()

        # Allow trades only in NORMAL volatility (not extreme spikes)
        vol_ok = df["vol_filter"] < VOL_THRESHOLD
    else:
        vol_ok = True

    # =========================
    # Initialize
    # =========================
    df["signal"] = 0

    # =========================
    # Long Condition (unchanged)
    # =========================
    long_condition = (
        trend_up &
        (df["rsi_14"] < SIGNAL_RSI_THRESHOLD) &
        vol_ok
    )

    df.loc[long_condition, "signal"] = 1

    # =========================
    # Short Condition (FIXED)
    # =========================
    if SIGNAL_USE_SHORT:
        short_condition = (
            trend_down &
            (df["rsi_14"] > 70) &   # true overbought
            vol_ok
        )
        df.loc[short_condition, "signal"] = -1

    # =========================
    # Labels
    # =========================
    df["signal_label"] = "Out of Market"
    df.loc[df["signal"] == 1, "signal_label"] = "Long"
    df.loc[df["signal"] == -1, "signal_label"] = "Short"

    return df