import numpy as np
import pandas as pd


def monte_carlo_forecast(
    df,
    days=252,
    simulations=1000,
    seed=42,
    use_fat_tails=True
):
    df = df.copy()

    returns = df["daily_return"].dropna()
    last_price = df["close"].dropna().iloc[-1]

    # Conservative drift assumption
    mu = min(returns.median(), 0.0005)
    sigma = returns.std()

    rng = np.random.default_rng(seed)

    if use_fat_tails:
        shocks = rng.standard_t(df=5, size=(simulations, days))
        shocks = shocks / np.std(shocks)
        simulated_returns = mu + sigma * shocks
    else:
        simulated_returns = rng.normal(mu, sigma, size=(simulations, days))

    paths = np.zeros((simulations, days + 1))
    paths[:, 0] = last_price

    for day in range(1, days + 1):
        paths[:, day] = paths[:, day - 1] * (1 + simulated_returns[:, day - 1])

    return paths


def garch_monte_carlo_forecast(
    df,
    garch_results,
    days=252,
    simulations=1000,
    seed=42
):
    df = df.copy()

    returns = df["daily_return"].dropna()
    last_price = df["close"].dropna().iloc[-1]

    # Conservative drift assumption
    mu = min(returns.median(), 0.0005)

    omega = garch_results["omega"]
    alpha = garch_results["alpha"]
    beta = garch_results["beta"]

    # Convert GARCH parameters from percent scale back to decimal scale
    omega = omega / 10000

    initial_vol = returns.std()
    initial_variance = initial_vol ** 2

    rng = np.random.default_rng(seed)

    paths = np.zeros((simulations, days + 1))
    paths[:, 0] = last_price

    variances = np.zeros((simulations, days + 1))
    variances[:, 0] = initial_variance

    for day in range(1, days + 1):
        shocks = rng.standard_t(df=5, size=simulations)
        shocks = shocks / np.std(shocks)

        vol = np.sqrt(variances[:, day - 1])
        simulated_returns = mu + vol * shocks

        paths[:, day] = paths[:, day - 1] * (1 + simulated_returns)

        variances[:, day] = (
            omega
            + alpha * (simulated_returns - mu) ** 2
            + beta * variances[:, day - 1]
        )

    return paths


def scenario_forecast(
    df,
    days=252,
    simulations=1000,
    seed=42
):
    df = df.copy()

    returns = df["daily_return"].dropna()
    last_price = df["close"].dropna().iloc[-1]
    historical_vol = returns.std()

    scenarios = {
        "Bear": {
            "daily_return": -0.0005,
            "daily_volatility": historical_vol * 1.25
        },
        "Base": {
            "daily_return": 0.0005,
            "daily_volatility": historical_vol
        },
        "Bull": {
            "daily_return": 0.0010,
            "daily_volatility": historical_vol * 0.90
        }
    }

    rng = np.random.default_rng(seed)
    results = {}

    for name, assumptions in scenarios.items():
        paths = np.zeros((simulations, days + 1))
        paths[:, 0] = last_price

        for day in range(1, days + 1):
            shocks = rng.standard_t(df=5, size=simulations)
            shocks = shocks / np.std(shocks)

            simulated_returns = (
                assumptions["daily_return"]
                + assumptions["daily_volatility"] * shocks
            )

            paths[:, day] = paths[:, day - 1] * (1 + simulated_returns)

        final_prices = paths[:, -1]

        results[name] = {
            "expected_price": np.mean(final_prices),
            "median_price": np.median(final_prices),
            "p5": np.percentile(final_prices, 5),
            "p25": np.percentile(final_prices, 25),
            "p75": np.percentile(final_prices, 75),
            "p95": np.percentile(final_prices, 95),
            "annualized_return_assumption": assumptions["daily_return"] * 252,
            "annualized_volatility_assumption": assumptions["daily_volatility"] * np.sqrt(252)
        }

    return results


def forecast_confidence_bands(paths):
    bands = pd.DataFrame({
        "day": range(paths.shape[1]),
        "p5": np.percentile(paths, 5, axis=0),
        "p25": np.percentile(paths, 25, axis=0),
        "median": np.percentile(paths, 50, axis=0),
        "p75": np.percentile(paths, 75, axis=0),
        "p95": np.percentile(paths, 95, axis=0)
    })

    return bands


def summarize_forecast(paths):
    final_prices = paths[:, -1]

    return {
        "expected_price": np.mean(final_prices),
        "median_price": np.median(final_prices),
        "p5": np.percentile(final_prices, 5),
        "p25": np.percentile(final_prices, 25),
        "p75": np.percentile(final_prices, 75),
        "p95": np.percentile(final_prices, 95)
    }