import numpy as np
import pandas as pd
from scipy.stats import norm


def calculate_var(returns, confidence_level=0.95):
    returns = returns.dropna()

    if len(returns) == 0:
        return np.nan

    return np.percentile(returns, (1 - confidence_level) * 100)


def calculate_parametric_var(returns, confidence_level=0.95):
    returns = returns.dropna()

    if len(returns) == 0:
        return np.nan

    mean = returns.mean()
    std = returns.std()

    z_score = norm.ppf(1 - confidence_level)

    return mean + z_score * std


def calculate_expected_shortfall(returns, confidence_level=0.95):
    returns = returns.dropna()

    if len(returns) == 0:
        return np.nan

    var_threshold = calculate_var(returns, confidence_level)
    tail_losses = returns[returns <= var_threshold]

    if len(tail_losses) == 0:
        return np.nan

    return tail_losses.mean()


def calculate_rolling_risk(df, window=126, confidence_level=0.95):
    df = df.copy()

    df["rolling_var"] = df["daily_return"].rolling(window).apply(
        lambda x: calculate_var(pd.Series(x), confidence_level),
        raw=False
    )

    df["rolling_expected_shortfall"] = df["daily_return"].rolling(window).apply(
        lambda x: calculate_expected_shortfall(pd.Series(x), confidence_level),
        raw=False
    )

    return df


def calculate_risk_models(df, confidence_level=0.95):
    returns = df["daily_return"].dropna()

    hist_var = calculate_var(returns, confidence_level)
    param_var = calculate_parametric_var(returns, confidence_level)
    es = calculate_expected_shortfall(returns, confidence_level)

    return {
        "confidence_level": confidence_level,
        "historical_var": hist_var,
        "parametric_var": param_var,
        "expected_shortfall": es
    }