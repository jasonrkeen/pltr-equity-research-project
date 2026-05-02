import numpy as np


def backtest_strategy(df):
    df = df.copy()

    # Avoid lookahead bias by using yesterday's signal
    df["strategy_return"] = df["signal"].shift(1) * df["daily_return"]
    df["strategy_return"] = df["strategy_return"].fillna(0)

    # Buy and hold return
    df["buy_hold_return"] = df["daily_return"].fillna(0)

    # Cumulative growth of $1
    df["strategy_cum"] = (1 + df["strategy_return"]).cumprod()
    df["buy_hold_cum"] = (1 + df["buy_hold_return"]).cumprod()

    return df


def calculate_sharpe(returns, rf=0):
    excess_returns = returns - rf
    volatility = excess_returns.std()

    if volatility == 0 or np.isnan(volatility):
        return np.nan

    return (excess_returns.mean() / volatility) * np.sqrt(252)


def calculate_performance(df):
    df = df.copy()
    df = df.dropna(subset=["strategy_return", "buy_hold_return"])

    # Dynamic risk-free rate if available
    if "rf_daily" in df.columns:
        rf = df["rf_daily"].ffill()
    else:
        rf = 0

    strategy_volatility = df["strategy_return"].std() * np.sqrt(252)
    buy_hold_volatility = df["buy_hold_return"].std() * np.sqrt(252)

    strategy_sharpe = calculate_sharpe(df["strategy_return"], rf)
    buy_hold_sharpe = calculate_sharpe(df["buy_hold_return"], rf)

    strategy_drawdown = df["strategy_cum"] / df["strategy_cum"].cummax() - 1
    buy_hold_drawdown = df["buy_hold_cum"] / df["buy_hold_cum"].cummax() - 1

    strategy_win_rate = (df["strategy_return"] > 0).mean()
    buy_hold_win_rate = (df["buy_hold_return"] > 0).mean()

    return {
        "strategy_volatility": strategy_volatility,
        "buy_hold_volatility": buy_hold_volatility,
        "strategy_sharpe": strategy_sharpe,
        "buy_hold_sharpe": buy_hold_sharpe,
        "strategy_max_drawdown": strategy_drawdown.min(),
        "buy_hold_max_drawdown": buy_hold_drawdown.min(),
        "strategy_win_rate": strategy_win_rate,
        "buy_hold_win_rate": buy_hold_win_rate,
        "strategy_total_return": df["strategy_cum"].iloc[-1] - 1,
        "buy_hold_total_return": df["buy_hold_cum"].iloc[-1] - 1
    }


def add_rolling_sharpe(df, window=126):
    df = df.copy()

    if "rf_daily" in df.columns:
        rf = df["rf_daily"].ffill()
    else:
        rf = 0

    df["strategy_excess_return"] = df["strategy_return"] - rf
    df["buy_hold_excess_return"] = df["buy_hold_return"] - rf

    strategy_rolling_std = df["strategy_excess_return"].rolling(window).std()
    buy_hold_rolling_std = df["buy_hold_excess_return"].rolling(window).std()

    df["strategy_rolling_sharpe"] = (
        df["strategy_excess_return"].rolling(window).mean()
        / strategy_rolling_std
    ) * np.sqrt(252)

    df["buy_hold_rolling_sharpe"] = (
        df["buy_hold_excess_return"].rolling(window).mean()
        / buy_hold_rolling_std
    ) * np.sqrt(252)

    # Keep the chart readable when rolling volatility is very low
    df["strategy_rolling_sharpe"] = df["strategy_rolling_sharpe"].replace([np.inf, -np.inf], np.nan).clip(-5, 5)
    df["buy_hold_rolling_sharpe"] = df["buy_hold_rolling_sharpe"].replace([np.inf, -np.inf], np.nan).clip(-5, 5)

    return df