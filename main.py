import argparse
import pandas as pd

from config.config import (
    TICKER, START, END,
    BENCHMARKS, PEERS, FACTOR_ETFS,
    RISK_FREE_TICKER,
    TRADING_DAYS, NUM_SIMULATIONS,
    SIGNAL_RSI_THRESHOLD,
    CLEAN_RUN
)

from src.data_manager import load_assets
from src.output_manager import (
    setup_project_folders,
    clean_outputs,
    create_timestamped_run_folder,
    copy_outputs_to_run_folder
)

from src.indicators import add_indicators
from src.signals import generate_signals
from src.backtest import backtest_strategy, calculate_performance, add_rolling_sharpe

from src.risk_metrics import calculate_risk
from src.risk_models import calculate_risk_models, calculate_rolling_risk

from src.factor_analysis import calculate_alpha_beta
from src.multi_factor import build_factor_dataframe, run_multi_factor_regression

from src.stat_tests import compare_strategy_vs_buy_hold, add_rolling_significance
from src.regime_detection import add_regimes, summarize_regimes

from src.forecast import (
    monte_carlo_forecast,
    garch_monte_carlo_forecast,
    scenario_forecast,
    forecast_confidence_bands,
    summarize_forecast
)

from src.garch_model import run_garch
from src.fundamentals import build_fundamental_summary
from src.government_exposure import build_government_exposure_summary
from src.peer_analysis import compute_metrics

from src.visualization import (
    plot_price,
    plot_strategy,
    plot_drawdown,
    plot_benchmark_comparison,
    plot_peer_comparison,
    plot_rolling_sharpe,
    plot_rolling_tail_risk,
    plot_rolling_significance,
    plot_monte_carlo,
    plot_forecast_bands,
    plot_scenario_forecast,
    plot_garch_volatility_forecast,
    plot_relative_performance,
)

from src.report_generator import generate_pdf_report

# =========================
# Helper Functions
# =========================
def compute_returns(data):
    data = data.copy()

    price_col = "adj_close" if "adj_close" in data.columns else "close"

    data["daily_return"] = data[price_col].pct_change()
    data["cum_return"] = (1 + data["daily_return"].fillna(0)).cumprod()

    return data


def compute_risk_free_rate(data):
    data = data.copy()

    price_col = "adj_close" if "adj_close" in data.columns else "close"

    # Treasury yield tickers are quoted in percentage terms
    data["rf_daily"] = (data[price_col] / 100) / 252

    return data[["date", "rf_daily"]]


def require_data(market_data, ticker):
    data = market_data.get(ticker)

    if data is None:
        raise ValueError(f"Missing data for {ticker}")

    return data


def generate_legacy_ma_signals(df):
    """
    Legacy Strategy v1:
    MA50 > MA200 plus RSI pullback filter.
    Long-only.
    """
    df = df.copy()

    df["signal"] = 0

    long_condition = (
        (df["ma50"] > df["ma200"]) &
        (df["rsi_14"] < SIGNAL_RSI_THRESHOLD)
    )

    df.loc[long_condition, "signal"] = 1

    df["signal_label"] = "Out of Market"
    df.loc[df["signal"] == 1, "signal_label"] = "Long"

    return df


def build_strategy_comparison(strategy_v1, strategy_v2):
    v1_perf = calculate_performance(strategy_v1)
    v2_perf = calculate_performance(strategy_v2)

    comparison = pd.DataFrame({
        "Strategy v1 - MA/RSI": {
            "total_return": v1_perf["strategy_total_return"],
            "sharpe": v1_perf["strategy_sharpe"],
            "max_drawdown": v1_perf["strategy_max_drawdown"],
            "win_rate": v1_perf.get("strategy_win_rate")
        },
        "Strategy v2 - Enhanced": {
            "total_return": v2_perf["strategy_total_return"],
            "sharpe": v2_perf["strategy_sharpe"],
            "max_drawdown": v2_perf["strategy_max_drawdown"],
            "win_rate": v2_perf.get("strategy_win_rate")
        },
        "Buy & Hold": {
            "total_return": v2_perf["buy_hold_total_return"],
            "sharpe": v2_perf["buy_hold_sharpe"],
            "max_drawdown": v2_perf["buy_hold_max_drawdown"],
            "win_rate": v2_perf.get("buy_hold_win_rate")
        }
    }).T

    return comparison


# =========================
# CLI Arguments
# =========================
parser = argparse.ArgumentParser(description="PLTR Equity Research Pipeline")

parser.add_argument(
    "--no-clean",
    action="store_true",
    help="Do not clean old output files before running"
)

parser.add_argument(
    "--archive-run",
    action="store_true",
    help="Archive generated outputs into a timestamped folder"
)

args = parser.parse_args()


# =========================
# Output Setup
# =========================
setup_project_folders()

if CLEAN_RUN:
    print("\nCleaning previous outputs...")
    clean_outputs(enabled=True)
else:
    print("\nSkipping clean run (CLEAN_RUN = False)")


# =========================
# LOAD DATA
# =========================
tickers = list(dict.fromkeys(
    [TICKER]
    + BENCHMARKS
    + PEERS
    + list(FACTOR_ETFS.values())
    + [RISK_FREE_TICKER]
))

market_data = load_assets(tickers, START, END)

df = require_data(market_data, TICKER)

spy = require_data(market_data, "SPY")
xlk = require_data(market_data, "XLK")
ita = require_data(market_data, "ITA")

msft = require_data(market_data, "MSFT")
snow = require_data(market_data, "SNOW")
ddog = require_data(market_data, "DDOG")
ibm = require_data(market_data, "IBM")

iwm = require_data(market_data, "IWM")
vlue = require_data(market_data, "VLUE")
mtum = require_data(market_data, "MTUM")

rf = require_data(market_data, RISK_FREE_TICKER)


# =========================
# PREPARE BENCHMARKS, PEERS, AND FACTORS
# =========================
spy = compute_returns(spy)
xlk = compute_returns(xlk)
ita = compute_returns(ita)

msft = compute_returns(msft)
snow = compute_returns(snow)
ddog = compute_returns(ddog)
ibm = compute_returns(ibm)

iwm = compute_returns(iwm)
vlue = compute_returns(vlue)
mtum = compute_returns(mtum)


# =========================
# PREPARE RISK-FREE RATE
# =========================
rf_daily = compute_risk_free_rate(rf)


# =========================
# PREPARE PLTR DATA
# =========================
df_base = add_indicators(df)

df_base = df_base.merge(
    rf_daily,
    on="date",
    how="left"
)

df_base["rf_daily"] = df_base["rf_daily"].ffill().bfill()


# =========================
# STRATEGY COMPARISON
# =========================
strategy_v1 = generate_legacy_ma_signals(df_base)
strategy_v1 = backtest_strategy(strategy_v1)

df = generate_signals(df_base)
df = backtest_strategy(df)

strategy_comparison = build_strategy_comparison(strategy_v1, df)
strategy_comparison.to_csv("outputs/tables/strategy_comparison.csv")

print("\nStrategy v1 vs Strategy v2 Comparison")
print("-------------------------------------")
print(strategy_comparison)

print("\nEnhanced Signal Check")
print("---------------------")
signal_cols = [
    col for col in ["ema_short", "ema_long", "vol_filter", "signal", "signal_label"]
    if col in df.columns
]
if signal_cols:
    print(df[signal_cols].tail())


# =========================
# ADD ROLLING / REGIME ANALYSIS
# =========================
df = add_rolling_sharpe(df)

# Remove early unstable rolling window observations
df = df.iloc[100:]

df = calculate_rolling_risk(df)
df = add_rolling_significance(df)
df = add_regimes(df)
df = df.dropna(subset=["strategy_cum", "buy_hold_cum"])


# =========================
# RISK & PERFORMANCE
# =========================
metrics = calculate_risk(df)
performance = calculate_performance(df)
risk_models = calculate_risk_models(df)

print("\nPLTR Risk Metrics")
print("-----------------")
print(f"Annualized Volatility: {metrics['volatility']:.2%}")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Cumulative Return: {metrics['cumulative_return']:.2%}")

if "max_drawdown" in metrics:
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")

print("\nStrategy Performance")
print("--------------------")
print(f"Strategy Total Return: {performance['strategy_total_return']:.2%}")
print(f"Buy & Hold Total Return: {performance['buy_hold_total_return']:.2%}")
print(f"Strategy Sharpe Ratio: {performance['strategy_sharpe']:.2f}")
print(f"Buy & Hold Sharpe Ratio: {performance['buy_hold_sharpe']:.2f}")
print(f"Strategy Max Drawdown: {performance['strategy_max_drawdown']:.2%}")
print(f"Buy & Hold Max Drawdown: {performance['buy_hold_max_drawdown']:.2%}")

if "strategy_win_rate" in performance and "buy_hold_win_rate" in performance:
    print(f"Strategy Win Rate: {performance['strategy_win_rate']:.2%}")
    print(f"Buy & Hold Win Rate: {performance['buy_hold_win_rate']:.2%}")

print("\nTail Risk Metrics")
print("-----------------")
print(f"Historical VaR 95%: {risk_models['historical_var']:.2%}")
print(f"Parametric VaR 95%: {risk_models['parametric_var']:.2%}")
print(f"Expected Shortfall 95%: {risk_models['expected_shortfall']:.2%}")


# =========================
# FACTOR ANALYSIS
# =========================
rf_mean = df["rf_daily"].mean()

alpha_spy, beta_spy, r2_spy = calculate_alpha_beta(
    df["daily_return"],
    spy["daily_return"],
    rf_mean
)

alpha_xlk, beta_xlk, r2_xlk = calculate_alpha_beta(
    df["daily_return"],
    xlk["daily_return"],
    rf_mean
)

print("\nRegression Alpha & Beta")
print("-----------------------")
print(f"Alpha vs SPY: {alpha_spy:.2%} | Beta: {beta_spy:.2f} | R²: {r2_spy:.2%}")
print(f"Alpha vs XLK: {alpha_xlk:.2%} | Beta: {beta_xlk:.2f} | R²: {r2_xlk:.2%}")

factor_df = build_factor_dataframe(df, spy, iwm, vlue, mtum, ita)
multi_factor_model = run_multi_factor_regression(factor_df)

factor_results = pd.DataFrame({
    "coefficient": multi_factor_model.params,
    "p_value": multi_factor_model.pvalues
})

factor_results.to_csv("outputs/tables/multi_factor_regression.csv")

print("\nMulti-Factor Regression")
print("-----------------------")
print(factor_results)


# =========================
# STAT TEST
# =========================
test_results = compare_strategy_vs_buy_hold(df)

print("\nStatistical Test")
print("----------------")
print(f"T-statistic: {test_results['t_stat']:.4f}")
print(f"P-value: {test_results['p_value']:.4f}")

if "mean_return_difference" in test_results:
    print(f"Mean Return Difference: {test_results['mean_return_difference']:.6f}")

if "annualized_return_difference" in test_results:
    print(f"Annualized Return Difference: {test_results['annualized_return_difference']:.2%}")

if "sharpe_difference" in test_results:
    print(f"Sharpe Difference: {test_results['sharpe_difference']:.2f}")

if "tracking_error" in test_results:
    print(f"Tracking Error: {test_results['tracking_error']:.2%}")

if "information_ratio" in test_results:
    print(f"Information Ratio: {test_results['information_ratio']:.2f}")


# =========================
# REGIMES
# =========================
regime_summary = summarize_regimes(df)
regime_summary.to_csv("outputs/tables/regime_summary.csv", index=False)

print("\nRegime Summary")
print("--------------")
print(regime_summary)


# =========================
# GARCH
# =========================
garch_results = run_garch(df)

if garch_results:
    print("\nGARCH Model")
    print("------------")
    print(f"Omega: {garch_results['omega']:.4f}")
    print(f"Alpha: {garch_results['alpha']:.4f}")
    print(f"Beta: {garch_results['beta']:.4f}")
    print(f"Persistence (α+β): {garch_results['persistence']:.4f}")

    if garch_results.get("long_run_volatility") is not None:
        print(f"Long-run Volatility: {garch_results['long_run_volatility']:.2%}")

    if "volatility_forecast" in garch_results:
        print("\nGARCH Volatility Forecast")
        print("-------------------------")
        print(garch_results["volatility_forecast"].tail())


# =========================
# FORECAST
# =========================
# Standard fat-tail Monte Carlo
paths = monte_carlo_forecast(
    df,
    days=TRADING_DAYS,
    simulations=NUM_SIMULATIONS,
    use_fat_tails=True
)

# GARCH + fat-tail Monte Carlo, if GARCH results are available
if garch_results:
    garch_paths = garch_monte_carlo_forecast(
        df,
        garch_results,
        days=TRADING_DAYS,
        simulations=NUM_SIMULATIONS
    )
else:
    garch_paths = paths

forecast_summary = summarize_forecast(garch_paths)
bands = forecast_confidence_bands(garch_paths)
scenarios = scenario_forecast(df)

bands.to_csv("outputs/tables/forecast_confidence_bands.csv", index=False)
pd.DataFrame(scenarios).T.to_csv("outputs/tables/scenario_forecast.csv")

print("\nForecast Summary")
print("----------------")
print(f"Expected Price: {forecast_summary['expected_price']:.2f}")
print(f"Median Price: {forecast_summary['median_price']:.2f}")
print(f"5th Percentile: {forecast_summary['p5']:.2f}")
print(f"95th Percentile: {forecast_summary['p95']:.2f}")

print("\nScenario Forecast")
print("-----------------")
print(pd.DataFrame(scenarios).T)

print("\nForecast Confidence Bands")
print("-------------------------")
print(bands.tail(1))


# =========================
# FUNDAMENTALS
# =========================
fundamentals = build_fundamental_summary()

print("\nFundamental Summary")
print("-------------------")
print(fundamentals.tail())


# =========================
# GOVERNMENT EXPOSURE
# =========================
gov = build_government_exposure_summary(df, ita)

print("\nGovernment & Defense Exposure")
print("-----------------------------")
print(gov)


# =========================
# PEER ANALYSIS
# =========================
peer_metrics = {
    "PLTR": compute_metrics(df, rf_mean),
    "MSFT": compute_metrics(msft, rf_mean),
    "SNOW": compute_metrics(snow, rf_mean),
    "DDOG": compute_metrics(ddog, rf_mean),
    "IBM": compute_metrics(ibm, rf_mean),
    "SPY": compute_metrics(spy, rf_mean),
    "XLK": compute_metrics(xlk, rf_mean),
    "ITA": compute_metrics(ita, rf_mean),
}

peer_df = pd.DataFrame(peer_metrics).T
peer_df.to_csv("outputs/tables/peer_comparison.csv")

print("\nPeer Comparison")
print("----------------")
print(peer_df)


# =========================
# SAVE PROCESSED DATA
# =========================
df.to_csv("data/processed/pltr_processed.csv", index=False)


# =========================
# VISUALS
# =========================
plot_price(df)
plot_strategy(df)
plot_drawdown(df)

plot_benchmark_comparison(df, spy, xlk, ita)
plot_peer_comparison(df, snow, ddog, ibm, msft)
plot_relative_performance(df, snow, ddog, ibm, msft)

plot_rolling_sharpe(df)
plot_rolling_tail_risk(df)
plot_rolling_significance(df)

plot_monte_carlo(garch_paths)
plot_forecast_bands(bands)
plot_scenario_forecast(scenarios)

plot_garch_volatility_forecast(garch_results)

report_path = generate_pdf_report(
    performance=performance,
    metrics=metrics,
    risk_models=risk_models,
    forecast_summary=forecast_summary,
    test_results=test_results
)
print(f"\nPDF report generated: {report_path}")

# =========================
# ARCHIVE RUN
# =========================
if args.archive_run:
    run_folder = create_timestamped_run_folder()
    copy_outputs_to_run_folder(run_folder)
    print(f"\nArchived outputs to: {run_folder}")