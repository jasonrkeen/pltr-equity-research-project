import numpy as np


def calculate_sharpe(returns, rf=0):
    excess_returns = returns - rf
    volatility = excess_returns.std()

    if volatility == 0 or np.isnan(volatility):
        return np.nan

    return excess_returns.mean() / volatility * np.sqrt(252)


def calculate_max_drawdown(cumulative):
    drawdown = cumulative / cumulative.cummax() - 1
    return drawdown.min()


def calculate_risk(df):
    df = df.copy()
    df = df.dropna(subset=["daily_return"])

    returns = df["daily_return"]

    if "rf_daily" in df.columns:
        rf = df["rf_daily"].ffill()
    else:
        rf = 0

    volatility = returns.std() * np.sqrt(252)
    sharpe = calculate_sharpe(returns, rf)

    if "cumulative_return" in df.columns:
        cumulative = df["cumulative_return"].dropna()
    else:
        cumulative = (1 + returns.fillna(0)).cumprod()

    cumulative_return = cumulative.iloc[-1] - 1
    max_drawdown = calculate_max_drawdown(cumulative)

    return {
        "volatility": volatility,
        "sharpe_ratio": sharpe,
        "cumulative_return": cumulative_return,
        "max_drawdown": max_drawdown
    }