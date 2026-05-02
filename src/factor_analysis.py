import numpy as np


def calculate_alpha_beta(asset_returns, benchmark_returns, risk_free_rate=0.04 / 252):
    """
    Calculate annualized alpha and beta relative to a benchmark.

    Parameters:
        asset_returns: Series of asset daily returns
        benchmark_returns: Series of benchmark daily returns
        risk_free_rate: Daily risk-free rate

    Returns:
        alpha: Annualized alpha
        beta: Benchmark beta
        r_squared: Fraction of asset return variance explained by benchmark
    """

    # Align asset and benchmark returns
    df = asset_returns.to_frame("asset").join(
        benchmark_returns.to_frame("benchmark"),
        how="inner"
    ).dropna()

    # Excess returns
    df["asset_excess"] = df["asset"] - risk_free_rate
    df["benchmark_excess"] = df["benchmark"] - risk_free_rate

    benchmark_variance = np.var(df["benchmark_excess"])

    if benchmark_variance == 0 or np.isnan(benchmark_variance):
        return np.nan, np.nan, np.nan

    # Beta
    beta = (
        np.cov(df["asset_excess"], df["benchmark_excess"])[0][1]
        / benchmark_variance
    )

    # Daily alpha
    daily_alpha = df["asset_excess"].mean() - beta * df["benchmark_excess"].mean()

    # Annualized alpha
    alpha = daily_alpha * 252

    # R-squared
    correlation = df["asset_excess"].corr(df["benchmark_excess"])
    r_squared = correlation ** 2

    return alpha, beta, r_squared