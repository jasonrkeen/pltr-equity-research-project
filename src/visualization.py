import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib.patches import Patch


# =========================
# Shared Helpers
# =========================
def _save_chart(filename, show=False):
    os.makedirs("outputs/charts", exist_ok=True)

    plt.tight_layout()
    plt.savefig(
        f"outputs/charts/{filename}",
        dpi=300,
        bbox_inches="tight",
        facecolor="white"
    )

    if show:
        plt.show()

    plt.close()


def _get_regime_patches():
    return [
        Patch(facecolor="green", alpha=0.05, label="Bull / Low Vol"),
        Patch(facecolor="green", alpha=0.10, label="Bull / High Vol"),
        Patch(facecolor="red", alpha=0.05, label="Bear / Low Vol"),
        Patch(facecolor="red", alpha=0.10, label="Bear / High Vol"),
    ]


def _shade_regime(df, start, end, regime):
    colors = {
        "Bull / Low Vol": ("green", 0.05),
        "Bull / High Vol": ("green", 0.10),
        "Bear / Low Vol": ("red", 0.05),
        "Bear / High Vol": ("red", 0.10),
    }

    if regime not in colors:
        return

    color, alpha = colors[regime]

    plt.axvspan(
        df["date"].iloc[start],
        df["date"].iloc[end],
        color=color,
        alpha=alpha,
        linewidth=0
    )


def _shade_and_label_regimes(df, show_labels=False):
    if "regime" not in df.columns or df.empty:
        return

    regimes = df["regime"].fillna("Neutral")

    start_idx = 0
    current_regime = regimes.iloc[0]

    for i in range(1, len(df)):
        if regimes.iloc[i] != current_regime:
            _shade_regime(df, start_idx, i, current_regime)

            if show_labels:
                _label_regime(df, start_idx, i, current_regime)

            start_idx = i
            current_regime = regimes.iloc[i]

    _shade_regime(df, start_idx, len(df) - 1, current_regime)

    if show_labels:
        _label_regime(df, start_idx, len(df) - 1, current_regime)


def _label_regime(df, start, end, regime):
    if regime == "Neutral":
        return

    # Only label very long regimes to avoid clutter
    if end - start < 120:
        return

    midpoint = start + (end - start) // 2

    label_map = {
        "Bull / Low Vol": "Bull\nLow Vol",
        "Bull / High Vol": "Bull\nHigh Vol",
        "Bear / Low Vol": "Bear\nLow Vol",
        "Bear / High Vol": "Bear\nHigh Vol",
    }

    label = label_map.get(regime)

    if not label:
        return

    ax = plt.gca()
    ymin, ymax = ax.get_ylim()
    y_pos = ymin + (ymax - ymin) * 0.08

    plt.text(
        df["date"].iloc[midpoint],
        y_pos,
        label,
        ha="center",
        va="bottom",
        fontsize=7,
        alpha=0.65
    )


# =========================
# Price / Strategy Charts
# =========================
def plot_price(df):
    df = df.copy()

    plt.figure(figsize=(12, 6))

    # --- Regime shading ---
    if "regime" in df.columns:
        _shade_and_label_regimes(df, show_labels=False)

    # --- Price + Moving Averages ---
    plt.plot(df["date"], df["close"], label="Close", linewidth=2)
    plt.plot(df["date"], df["ma50"], label="MA50", linestyle="--", alpha=0.7)
    plt.plot(df["date"], df["ma200"], label="MA200", linestyle="--", alpha=0.7)

    # --- Signals ---
    if "signal" in df.columns:
        buy_signals = df[df["signal"] == 1]
        sell_signals = df[df["signal"] == -1]

        plt.scatter(
            buy_signals["date"],
            buy_signals["close"],
            marker="^",
            color="green",
            edgecolor="black",
            linewidth=0.5,
            label="Long Signal",
            alpha=0.9,
            zorder=5
        )

        plt.scatter(
            sell_signals["date"],
            sell_signals["close"],
            marker="v",
            color="red",
            edgecolor="black",
            linewidth=0.5,
            label="Short Signal",
            alpha=0.9,
            zorder=5
        )

    plt.ylim(bottom=0)

    plt.title("PLTR Price with Signals and Market Regimes")
    plt.xlabel("Date")
    plt.ylabel("Price")

    plt.legend(
        loc="upper left",
        fontsize=10,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.9
    )

    plt.grid(True, linestyle="--", alpha=0.3)
    plt.gcf().autofmt_xdate()

    _save_chart("pltr_price_ma.png")


def plot_strategy(df):
    df = df.copy()

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # --- Regime shading ---
    if "regime" in df.columns:
        plt.sca(ax1)
        _shade_and_label_regimes(df, show_labels=False)

    # --- Strategy vs Buy & Hold ---
    ax1.plot(
        df["date"],
        df["strategy_cum"],
        label="Enhanced Strategy",
        linewidth=2,
        color="blue"
    )

    ax1.plot(
        df["date"],
        df["buy_hold_cum"],
        label="Buy & Hold",
        linewidth=2,
        color="orange",
        alpha=0.85
    )

    ax1.axhline(
        1,
        linestyle="--",
        linewidth=1,
        color="black",
        label="$1 Starting Value"
    )

    ax1.set_yscale("log")
    ax1.set_ylim(bottom=0.5)

    # --- Secondary axis: outperformance ---
    outperformance = df["strategy_cum"] / df["buy_hold_cum"].replace(0, float("nan"))

    ax2 = ax1.twinx()

    ax2.plot(
        df["date"],
        outperformance,
        linestyle=":",
        linewidth=2,
        color="purple",
        label="Outperformance"
    )

    ax2.axhline(
        1,
        linestyle="--",
        linewidth=1,
        color="purple",
        alpha=0.6
    )

    ax1.set_title("PLTR Strategy vs Buy & Hold with Outperformance")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Growth of $1")
    ax2.set_ylabel("Strategy / Buy & Hold")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()

    ax1.legend(
        lines_1 + lines_2,
        labels_1 + labels_2,
        loc="upper left",
        fontsize=10,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.9
    )

    ax1.grid(True, which="both", linestyle="--", alpha=0.3)
    fig.autofmt_xdate()

    _save_chart("pltr_strategy_vs_buyhold.png")


def plot_drawdown(df):
    df = df.copy()

    strategy_dd = df["strategy_cum"] / df["strategy_cum"].cummax() - 1
    buy_hold_dd = df["buy_hold_cum"] / df["buy_hold_cum"].cummax() - 1

    plt.figure(figsize=(12, 6))

    if "regime" in df.columns:
        _shade_and_label_regimes(df, show_labels=False)

    plt.plot(
        df["date"],
        strategy_dd,
        label="Strategy Drawdown",
        color="blue",
        linewidth=2
    )

    plt.plot(
        df["date"],
        buy_hold_dd,
        label="Buy & Hold Drawdown",
        color="orange",
        linewidth=2,
        alpha=0.85
    )

    plt.fill_between(df["date"], strategy_dd, 0, color="blue", alpha=0.08)
    plt.fill_between(df["date"], buy_hold_dd, 0, color="orange", alpha=0.10)

    if "signal" in df.columns:
        buy_signals = df[df["signal"] == 1]
        sell_signals = df[df["signal"] == -1]

        plt.scatter(
            buy_signals["date"],
            strategy_dd.loc[buy_signals.index],
            marker="^",
            color="green",
            edgecolor="black",
            linewidth=0.5,
            label="Long Signal",
            alpha=0.8,
            zorder=5
        )

        plt.scatter(
            sell_signals["date"],
            strategy_dd.loc[sell_signals.index],
            marker="v",
            color="red",
            edgecolor="black",
            linewidth=0.5,
            label="Short Signal",
            alpha=0.8,
            zorder=5
        )

    plt.axhline(0, linestyle="--", linewidth=1, color="black")

    plt.title("PLTR Drawdown with Market Regimes and Signals")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))

    plt.legend(
        fontsize=10,
        loc="lower left",
        bbox_to_anchor=(0.01, 0.02),
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.9
    )

    plt.grid(True, linestyle="--", alpha=0.3)
    plt.gcf().autofmt_xdate()

    _save_chart("pltr_drawdown.png")

    # =========================
    # LinkedIn Hero Chart Export
    # =========================
    plt.figure(figsize=(14, 7.5), facecolor="white")

    if "regime" in df.columns:
        _shade_and_label_regimes(df, show_labels=False)

    plt.plot(
        df["date"],
        strategy_dd,
        label="Risk-Managed Strategy",
        color="#1f77b4",
        linewidth=3
    )

    plt.plot(
        df["date"],
        buy_hold_dd,
        label="Buy & Hold",
        color="#ff7f0e",
        linewidth=3
    )

    plt.fill_between(df["date"], strategy_dd, 0, color="#1f77b4", alpha=0.04)
    plt.fill_between(df["date"], buy_hold_dd, 0, color="#ff7f0e", alpha=0.08)

    plt.axhline(0, linestyle="--", linewidth=1, color="black", alpha=0.7)

    plt.title(
        "PLTR Risk Reduction: Strategy vs Buy & Hold\n"
        "Max Drawdown Improved from -79.14% to -33.14%",
        fontsize=17,
        fontweight="bold",
        pad=12
    )

    plt.xlabel("Date", fontsize=11)
    plt.ylabel("Drawdown", fontsize=11)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))

    plt.legend(
        fontsize=11,
        loc="lower left",
        bbox_to_anchor=(0.01, 0.02),
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.9
    )

    plt.grid(True, linestyle="--", alpha=0.20)
    plt.gca().set_facecolor("white")
    plt.gcf().autofmt_xdate()

    plt.tight_layout()
    plt.savefig(
        "outputs/charts/linkedin_drawdown.png",
        dpi=400,
        bbox_inches="tight",
        facecolor="white"
    )
    plt.close()

# =========================
# Benchmark / Peer Charts
# =========================
def plot_benchmark_comparison(df, spy, xlk, ita):
    plt.figure(figsize=(12, 6))

    plt.plot(
        df["date"],
        df["buy_hold_cum"],
        label="PLTR",
        linewidth=2.75,
        color="black"
    )

    plt.plot(spy["date"], spy["cum_return"], label="SPY", linewidth=1.8, alpha=0.8)
    plt.plot(xlk["date"], xlk["cum_return"], label="XLK", linewidth=1.8, alpha=0.8)
    plt.plot(
        ita["date"],
        ita["cum_return"],
        label="ITA (Aerospace & Defense)",
        linewidth=1.8,
        alpha=0.8
    )

    plt.yscale("log")

    plt.title("PLTR vs Market, Technology, and Aerospace & Defense Benchmarks")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1 (Log Scale)")

    plt.legend(
        fontsize=10,
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.9
    )

    plt.grid(True, which="both", linestyle="--", alpha=0.3)
    plt.gcf().autofmt_xdate()

    _save_chart("pltr_vs_benchmarks.png")


def plot_peer_comparison(df, snow, ddog, ibm, msft):
    plt.figure(figsize=(12, 6))

    def normalize(series):
        series = series.dropna()

        if series.empty or series.iloc[0] == 0:
            return series

        return series / series.iloc[0]

    pltr = normalize(df["buy_hold_cum"])
    msft_norm = normalize(msft["cum_return"])
    snow_norm = normalize(snow["cum_return"])
    ddog_norm = normalize(ddog["cum_return"])
    ibm_norm = normalize(ibm["cum_return"])

    plt.plot(df["date"], pltr, label="PLTR", linewidth=2.75, color="black")
    plt.plot(msft["date"], msft_norm, label="MSFT", linewidth=1.8, alpha=0.75)
    plt.plot(snow["date"], snow_norm, label="SNOW", linewidth=1.8, alpha=0.75)
    plt.plot(ddog["date"], ddog_norm, label="DDOG", linewidth=1.8, alpha=0.75)
    plt.plot(ibm["date"], ibm_norm, label="IBM", linewidth=1.8, alpha=0.75)

    plt.yscale("log")

    plt.title("PLTR vs Peer Companies (Normalized Growth, Log Scale)")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1 (Log Scale)")

    plt.legend(
        fontsize=10,
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.9
    )

    plt.grid(True, which="both", linestyle="--", alpha=0.3)
    plt.gcf().autofmt_xdate()

    _save_chart("pltr_peer_comparison.png")


def plot_relative_performance(df, snow, ddog, ibm, msft):
    plt.figure(figsize=(12, 6))

    base = df[["date", "buy_hold_cum"]].rename(
        columns={"buy_hold_cum": "PLTR"}
    )

    peers = {
        "MSFT": msft,
        "SNOW": snow,
        "DDOG": ddog,
        "IBM": ibm,
    }

    relative_df = base.copy()

    for ticker, peer_df in peers.items():
        temp = peer_df[["date", "cum_return"]].rename(
            columns={"cum_return": ticker}
        )

        relative_df = relative_df.merge(temp, on="date", how="inner")

    for ticker in peers.keys():
        relative_df[f"PLTR_vs_{ticker}"] = (
            relative_df["PLTR"] / relative_df[ticker].replace(0, float("nan"))
        )

        plt.plot(
            relative_df["date"],
            relative_df[f"PLTR_vs_{ticker}"],
            label=f"PLTR / {ticker}",
            linewidth=1.8,
            alpha=0.85
        )

    plt.axhline(
        1,
        linestyle="--",
        linewidth=1,
        color="black",
        alpha=0.8,
        label="Equal Performance"
    )

    plt.title("PLTR Relative Performance vs Peers")
    plt.xlabel("Date")
    plt.ylabel("Relative Performance Ratio")

    plt.legend(
        fontsize=10,
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.9
    )

    plt.grid(True, linestyle="--", alpha=0.3)
    plt.gcf().autofmt_xdate()

    _save_chart("pltr_relative_performance.png")


# =========================
# Rolling Risk / Statistical Charts
# =========================
def plot_rolling_sharpe(df):
    chart_df = df.copy()

    chart_df["strategy_rolling_sharpe"] = chart_df["strategy_rolling_sharpe"].clip(-5, 5)
    chart_df["buy_hold_rolling_sharpe"] = chart_df["buy_hold_rolling_sharpe"].clip(-5, 5)

    plt.figure(figsize=(12, 6))

    plt.plot(
        chart_df["date"],
        chart_df["strategy_rolling_sharpe"],
        label="Strategy Rolling Sharpe",
        linewidth=2
    )

    plt.plot(
        chart_df["date"],
        chart_df["buy_hold_rolling_sharpe"],
        label="Buy & Hold Rolling Sharpe",
        linewidth=2,
        alpha=0.85
    )

    plt.axhline(0, linestyle="--", linewidth=1, color="black", alpha=0.8)

    plt.title("Rolling Sharpe Ratio: Strategy vs Buy & Hold")
    plt.xlabel("Date")
    plt.ylabel("Rolling Sharpe Ratio")

    plt.legend(
        fontsize=10,
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.9
    )

    plt.grid(True, linestyle="--", alpha=0.3)
    plt.gcf().autofmt_xdate()

    _save_chart("pltr_rolling_sharpe.png")


def plot_rolling_tail_risk(df):
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(
        df["date"],
        df["rolling_var"],
        label="Rolling VaR 95%",
        linewidth=2
    )

    ax1.plot(
        df["date"],
        df["rolling_expected_shortfall"],
        label="Rolling Expected Shortfall 95%",
        linewidth=2,
        alpha=0.85
    )

    ax1.axhline(0, linestyle="--", linewidth=1, color="black", alpha=0.8)

    ax1.set_title("Rolling Tail Risk: VaR and Expected Shortfall")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Daily Tail Loss")
    ax1.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax1.grid(True, linestyle="--", alpha=0.3)

    if "volatility_30" in df.columns:
        ax2 = ax1.twinx()

        ax2.plot(
            df["date"],
            df["volatility_30"],
            alpha=0.25,
            linestyle=":",
            linewidth=2,
            label="Rolling Volatility"
        )

        ax2.set_ylabel("Rolling Volatility")
        ax2.yaxis.set_major_formatter(PercentFormatter(1.0))

        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()

        ax1.legend(
            lines_1 + lines_2,
            labels_1 + labels_2,
            fontsize=10,
            loc="upper left",
            frameon=True,
            facecolor="white",
            edgecolor="none",
            framealpha=0.9
        )
    else:
        ax1.legend(
            fontsize=10,
            loc="upper left",
            frameon=True,
            facecolor="white",
            edgecolor="none",
            framealpha=0.9
        )

    fig.autofmt_xdate()

    _save_chart("pltr_rolling_tail_risk.png")


def plot_rolling_significance(df):
    plt.figure(figsize=(12, 6))

    plt.plot(
        df["date"],
        df["rolling_t_stat"],
        label="Rolling T-statistic",
        linewidth=2
    )

    plt.axhline(
        1.96,
        linestyle="--",
        color="black",
        linewidth=1,
        alpha=0.8,
        label="5% Significance Threshold"
    )

    plt.axhline(
        -1.96,
        linestyle="--",
        color="black",
        linewidth=1,
        alpha=0.8
    )

    plt.title("Rolling Statistical Significance: Strategy vs Buy & Hold")
    plt.xlabel("Date")
    plt.ylabel("Rolling T-statistic")

    plt.legend(
        fontsize=10,
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.9
    )

    plt.grid(True, linestyle="--", alpha=0.3)
    plt.gcf().autofmt_xdate()

    _save_chart("pltr_rolling_significance.png")


# =========================
# Forecast Charts
# =========================
def plot_monte_carlo(paths):
    plt.figure(figsize=(12, 6))

    for i in range(min(50, len(paths))):
        plt.plot(paths[i], alpha=0.14, linewidth=1)

    median_path = np.median(paths, axis=0)

    plt.plot(
        median_path,
        linewidth=2.75,
        linestyle="--",
        color="black",
        label="Median Path"
    )

    plt.title("Monte Carlo Simulation: Distribution of Future Price Paths")
    plt.xlabel("Days")
    plt.ylabel("Price")

    plt.legend(
        fontsize=10,
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.9
    )

    plt.grid(True, linestyle="--", alpha=0.3)

    _save_chart("monte_carlo.png")


def plot_forecast_bands(bands):
    plt.figure(figsize=(12, 6))

    current_price = bands["median"].iloc[0]

    plt.plot(
        bands["day"],
        bands["median"],
        label="Median Forecast",
        linewidth=2.5
    )

    plt.fill_between(
        bands["day"],
        bands["p25"],
        bands["p75"],
        alpha=0.30,
        label="25th-75th Percentile"
    )

    plt.fill_between(
        bands["day"],
        bands["p5"],
        bands["p95"],
        alpha=0.15,
        label="5th-95th Percentile"
    )

    plt.axhline(
        current_price,
        linestyle="--",
        color="black",
        linewidth=1,
        alpha=0.7,
        label="Current Price"
    )

    plt.title("Forecast Distribution: Median and Confidence Intervals")
    plt.xlabel("Days")
    plt.ylabel("Forecast Price")

    plt.legend(
        fontsize=10,
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.9
    )

    plt.grid(True, linestyle="--", alpha=0.3)

    _save_chart("pltr_forecast_bands.png")


def plot_scenario_forecast(scenarios):
    names = list(scenarios.keys())
    prices = [scenarios[name]["median_price"] for name in names]

    plt.figure(figsize=(10, 6))

    plt.bar(
        names,
        prices,
        alpha=0.85
    )

    plt.title("PLTR Scenario Forecast: Bear / Base / Bull")
    plt.xlabel("Scenario")
    plt.ylabel("Median Estimated Ending Price")
    plt.grid(True, axis="y", linestyle="--", alpha=0.3)

    _save_chart("pltr_scenario_forecast.png")


def plot_garch_volatility_forecast(garch_results):
    if not garch_results or "volatility_forecast" not in garch_results:
        return

    forecast = garch_results["volatility_forecast"]

    plt.figure(figsize=(12, 6))

    plt.plot(
        forecast["day"],
        forecast["forecast_volatility"],
        label="Forecast Volatility",
        linewidth=2.5
    )

    plt.title("GARCH Forecasted Volatility")
    plt.xlabel("Forecast Day")
    plt.ylabel("Daily Volatility")
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))

    plt.legend(
        fontsize=10,
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.9
    )

    plt.grid(True, linestyle="--", alpha=0.3)

    _save_chart("garch_volatility_forecast.png")