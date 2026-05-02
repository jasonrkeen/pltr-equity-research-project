def run_garch(df, forecast_horizon=30):
    try:
        from arch import arch_model
        import numpy as np
        import pandas as pd
        import os

        returns = df["daily_return"].dropna()

        if len(returns) < 100:
            raise ValueError("Not enough data for GARCH estimation")

        # Convert returns to percentages for arch package
        returns_pct = returns * 100

        # GARCH(1,1) with constant mean
        model = arch_model(
            returns_pct,
            vol="Garch",
            p=1,
            q=1,
            mean="Constant"
        )

        result = model.fit(disp="off")

        omega = result.params.get("omega")
        alpha = result.params.get("alpha[1]")
        beta = result.params.get("beta[1]")

        persistence = alpha + beta if alpha is not None and beta is not None else None

        if persistence is not None and persistence < 1:
            long_run_variance = omega / (1 - persistence)
            long_run_volatility = np.sqrt(long_run_variance) / 100
        else:
            long_run_volatility = None

        # Forecast future volatility
        forecast = result.forecast(horizon=forecast_horizon)
        variance_forecast = forecast.variance.iloc[-1]

        volatility_forecast = (variance_forecast ** 0.5) / 100

        volatility_forecast_df = pd.DataFrame({
            "day": range(1, forecast_horizon + 1),
            "forecast_volatility": volatility_forecast.values
        })

        os.makedirs("outputs/tables", exist_ok=True)
        volatility_forecast_df.to_csv(
            "outputs/tables/garch_volatility_forecast.csv",
            index=False
        )

        return {
            "omega": omega,
            "alpha": alpha,
            "beta": beta,
            "persistence": persistence,
            "long_run_volatility": long_run_volatility,
            "volatility_forecast": volatility_forecast_df
        }

    except Exception as e:
        print("\nGARCH not available:", e)
        return None