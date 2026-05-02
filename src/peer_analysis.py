import numpy as np


def get_cumulative_return_series(df):
    if "cum_return" in df.columns:
        return df["cum_return"].dropna()

    if "buy_hold_cum" in df.columns:
        return df["buy_hold_cum"].dropna()

    if "cumulative_return" in df.columns:
        return df["cumulative_return"].dropna()

    returns = df["daily_return"].fillna(0)
    return (1 + returns).cumprod()


def calculate_sharpe(returns, rf=0):
    excess_returns = returns - rf
    volatility = excess_returns.std()

    if volatility == 0 or np.isnan(volatility):
        return np.nan

    return excess_returns.mean() / volatility * np.sqrt(252)


def compute_metrics(df, risk_free_rate=0):
    returns = df["daily_return"].dropna()

    if returns.empty:
        return {
            "total_return": np.nan,
            "volatility": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "win_rate": np.nan
        }

    cumulative = get_cumulative_return_series(df)

    total_return = cumulative.iloc[-1] - 1
    volatility = returns.std() * np.sqrt(252)
    sharpe = calculate_sharpe(returns, risk_free_rate)

    drawdown = cumulative / cumulative.cummax() - 1
    max_drawdown = drawdown.min()

    win_rate = (returns > 0).mean()

    return {
        "total_return": total_return,
        "volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate
    }


def format_peer_metrics(peer_df):
    formatted = peer_df.copy()

    pct_cols = ["total_return", "volatility", "max_drawdown", "win_rate"]

    for col in pct_cols:
        if col in formatted.columns:
            formatted[col] = formatted[col].map(
                lambda x: f"{x:.2%}" if not np.isnan(x) else ""
            )

    if "sharpe" in formatted.columns:
        formatted["sharpe"] = formatted["sharpe"].map(
            lambda x: f"{x:.2f}" if not np.isnan(x) else ""
        )

    return formatted