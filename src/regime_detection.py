import numpy as np


def add_regimes(df, vol_window=30):
    df = df.copy()

    df["rolling_volatility"] = (
        df["daily_return"]
        .rolling(vol_window)
        .std()
        * np.sqrt(252)
    )

    vol_threshold = df["rolling_volatility"].median()

    df["regime"] = "Neutral"

    bull = df["ma50"] > df["ma200"]
    bear = df["ma50"] <= df["ma200"]
    low_vol = df["rolling_volatility"] <= vol_threshold
    high_vol = df["rolling_volatility"] > vol_threshold

    df.loc[bull & low_vol, "regime"] = "Bull / Low Vol"
    df.loc[bull & high_vol, "regime"] = "Bull / High Vol"
    df.loc[bear & low_vol, "regime"] = "Bear / Low Vol"
    df.loc[bear & high_vol, "regime"] = "Bear / High Vol"

    return df


def calculate_sharpe(returns):
    returns = returns.dropna()
    volatility = returns.std()

    if volatility == 0 or np.isnan(volatility):
        return np.nan

    return returns.mean() / volatility * np.sqrt(252)


def summarize_regimes(df):
    df = df.copy()

    regime_summary = (
        df.groupby("regime")
        .agg(
            avg_daily_return=("daily_return", "mean"),
            volatility=("daily_return", "std"),
            count=("daily_return", "count"),
            strategy_avg_return=("strategy_return", "mean"),
            buy_hold_avg_return=("buy_hold_return", "mean"),
        )
        .reset_index()
    )

    total_count = regime_summary["count"].sum()
    regime_summary["frequency"] = regime_summary["count"] / total_count

    regime_summary["annualized_return"] = (
        regime_summary["avg_daily_return"] * 252
    )

    regime_summary["annualized_volatility"] = (
        regime_summary["volatility"] * np.sqrt(252)
    )

    regime_summary["strategy_annualized_return"] = (
        regime_summary["strategy_avg_return"] * 252
    )

    regime_summary["buy_hold_annualized_return"] = (
        regime_summary["buy_hold_avg_return"] * 252
    )

    regime_summary["regime_sharpe"] = df.groupby("regime")["daily_return"].apply(
        calculate_sharpe
    ).values

    regime_summary["strategy_regime_sharpe"] = df.groupby("regime")["strategy_return"].apply(
        calculate_sharpe
    ).values

    regime_summary["buy_hold_regime_sharpe"] = df.groupby("regime")["buy_hold_return"].apply(
        calculate_sharpe
    ).values

    return regime_summary.sort_values("annualized_return", ascending=False)