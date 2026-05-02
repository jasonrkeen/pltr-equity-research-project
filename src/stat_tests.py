from scipy import stats
import numpy as np
import pandas as pd


def safe_sharpe(returns):
    returns = returns.dropna()
    volatility = returns.std()

    if volatility == 0 or np.isnan(volatility):
        return np.nan

    return returns.mean() / volatility * np.sqrt(252)


def compare_strategy_vs_buy_hold(df):
    clean = df[["strategy_return", "buy_hold_return"]].dropna()

    diff = clean["strategy_return"] - clean["buy_hold_return"]

    t_stat, p_value = stats.ttest_rel(
        clean["strategy_return"],
        clean["buy_hold_return"]
    )

    mean_diff = diff.mean()

    strategy_sharpe = safe_sharpe(clean["strategy_return"])
    buy_hold_sharpe = safe_sharpe(clean["buy_hold_return"])
    sharpe_diff = strategy_sharpe - buy_hold_sharpe

    tracking_error = diff.std() * np.sqrt(252)

    if tracking_error == 0 or np.isnan(tracking_error):
        information_ratio = np.nan
    else:
        information_ratio = (diff.mean() * 252) / tracking_error

    return {
        "t_stat": t_stat,
        "p_value": p_value,
        "mean_return_difference": mean_diff,
        "annualized_return_difference": mean_diff * 252,
        "sharpe_difference": sharpe_diff,
        "tracking_error": tracking_error,
        "information_ratio": information_ratio
    }


def add_rolling_significance(df, window=126):
    df = df.copy()

    df["return_difference"] = df["strategy_return"] - df["buy_hold_return"]

    def rolling_t_stat(x):
        x = pd.Series(x).dropna()

        if len(x) < 30 or x.std() == 0:
            return np.nan

        return x.mean() / (x.std() / np.sqrt(len(x)))

    def rolling_p_value(x):
        x = pd.Series(x).dropna()

        if len(x) < 30 or x.std() == 0:
            return np.nan

        t_stat = x.mean() / (x.std() / np.sqrt(len(x)))
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(x) - 1))

        return p_value

    def rolling_information_ratio(x):
        x = pd.Series(x).dropna()

        if len(x) < 30:
            return np.nan

        tracking_error = x.std() * np.sqrt(252)

        if tracking_error == 0 or np.isnan(tracking_error):
            return np.nan

        return (x.mean() * 252) / tracking_error

    df["rolling_t_stat"] = (
        df["return_difference"]
        .rolling(window)
        .apply(rolling_t_stat, raw=False)
    )

    df["rolling_p_value"] = (
        df["return_difference"]
        .rolling(window)
        .apply(rolling_p_value, raw=False)
    )

    df["rolling_information_ratio"] = (
        df["return_difference"]
        .rolling(window)
        .apply(rolling_information_ratio, raw=False)
    )

    return df