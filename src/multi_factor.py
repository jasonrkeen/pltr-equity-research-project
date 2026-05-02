import pandas as pd
import statsmodels.api as sm


def build_factor_dataframe(asset, spy, iwm, vlue, mtum, ita):
    factors = pd.DataFrame({
        "date": asset["date"],
        "asset_return": asset["daily_return"]
    })

    factor_data = (
        spy[["date", "daily_return"]]
        .rename(columns={"daily_return": "market"})
        .merge(
            iwm[["date", "daily_return"]].rename(columns={"daily_return": "size"}),
            on="date",
            how="inner"
        )
        .merge(
            vlue[["date", "daily_return"]].rename(columns={"daily_return": "value"}),
            on="date",
            how="inner"
        )
        .merge(
            mtum[["date", "daily_return"]].rename(columns={"daily_return": "momentum"}),
            on="date",
            how="inner"
        )
        .merge(
            ita[["date", "daily_return"]].rename(columns={"daily_return": "defense"}),
            on="date",
            how="inner"
        )
    )

    factors = factors.merge(factor_data, on="date", how="inner").dropna()

    return factors


def run_multi_factor_regression(factors):
    y = factors["asset_return"]

    X = factors[[
        "market",
        "size",
        "value",
        "momentum",
        "defense"
    ]]

    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()

    return model