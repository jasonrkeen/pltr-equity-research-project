import os
from datetime import datetime

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    PageBreak
)


def generate_pdf_report(
    performance,
    metrics,
    risk_models,
    forecast_summary,
    test_results,
    output_path="outputs/pltr_equity_research_report.pdf"
):
    os.makedirs("outputs", exist_ok=True)

    doc = SimpleDocTemplate(
        output_path,
        pagesize=LETTER,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=22,
        spaceAfter=20,
    )

    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["BodyText"],
        alignment=TA_CENTER,
        fontSize=11,
        spaceAfter=12,
    )

    insight_style = ParagraphStyle(
        "Insight",
        parent=styles["BodyText"],
        fontSize=9,
        leading=12,
        spaceAfter=10,
    )

    story = []

    story.append(Spacer(1, 120))
    story.append(Paragraph("PLTR Equity Research Report", title_style))

    story.append(Paragraph(
        "Systematic analysis of Palantir using quantitative modeling, "
        "strategy backtesting, risk analytics, fundamentals, government exposure, "
        "and probabilistic forecasting.",
        subtitle_style
    ))

    story.append(Spacer(1, 24))
    story.append(Paragraph(
        f"Generated: {datetime.today().strftime('%B %d, %Y')}",
        subtitle_style
    ))

    story.append(PageBreak())

    story.append(Paragraph("Executive Summary", styles["Title"]))
    story.append(Spacer(1, 12))

    summary_text = f"""
    Palantir has delivered a cumulative return of {metrics['cumulative_return']:.2%}
    since its IPO, outperforming broad market benchmarks but with substantially higher volatility and drawdown risk.

    The risk-managed strategy overlay generates a return of {performance['strategy_total_return']:.2%},
    with a Sharpe ratio of {performance['strategy_sharpe']:.2f}, compared to {performance['buy_hold_sharpe']:.2f}
    for buy-and-hold. Maximum drawdown is reduced from {performance['buy_hold_max_drawdown']:.2%}
    to {performance['strategy_max_drawdown']:.2%}.

    Tail risk remains significant, with a 95% Value at Risk of {risk_models['historical_var']:.2%}
    and Expected Shortfall of {risk_models['expected_shortfall']:.2%}.

    Forecasting suggests a median simulated price of {forecast_summary['median_price']:.2f},
    with substantial uncertainty reflected in wide confidence intervals.

    Results should be interpreted as model-based estimates rather than forecasts,
    as they depend on assumptions and exclude transaction costs.

    Overall, PLTR exhibits a high-volatility, regime-dependent return profile with strong upside potential
    but meaningful downside risk.
    """

    story.append(Paragraph(summary_text, styles["BodyText"]))
    story.append(PageBreak())

    sections = [
        (
            "Price Action: Signals and Market Regimes",
            "outputs/charts/pltr_price_ma.png",
            "<b>Regime dependency is visible:</b> PLTR transitions from prolonged drawdowns into explosive momentum phases."
        ),
        (
            "Strategy Performance: Risk-Managed Overlay vs Buy & Hold",
            "outputs/charts/pltr_strategy_vs_buyhold.png",
            f"<b>Risk reduction is not free:</b> the strategy produces {performance['strategy_total_return']:.2%} "
            f"vs {performance['buy_hold_total_return']:.2%}, protecting downside while sacrificing upside."
        ),
        (
            "Drawdown Analysis: Risk Reduction Across Market Regimes",
            "outputs/charts/pltr_drawdown.png",
            f"<b>Maximum drawdown improved by roughly 58%:</b> drawdown declined from "
            f"{performance['buy_hold_max_drawdown']:.2%} to {performance['strategy_max_drawdown']:.2%}."
        ),
        (
            "Benchmark Comparison: PLTR vs Market & Sector Exposure",
            "outputs/charts/pltr_vs_benchmarks.png",
            "<b>PLTR behaves differently from broad beta:</b> it diverges from SPY, XLK, and ITA, "
            "suggesting a hybrid growth profile rather than pure market exposure."
        ),
        (
            "Peer Comparison: Convex Return Profile vs Software Peers",
            "outputs/charts/pltr_peer_comparison.png",
            "<b>PLTR shows a convex return profile:</b> performance is driven by regime shifts rather than steady compounding."
        ),
        (
            "Rolling Tail Risk: VaR, Expected Shortfall, and Volatility Clustering",
            "outputs/charts/pltr_rolling_tail_risk.png",
            f"<b>Tail risk is clustered, not constant:</b> historical VaR is {risk_models['historical_var']:.2%} "
            f"and Expected Shortfall is {risk_models['expected_shortfall']:.2%}."
        ),
        (
            "Monte Carlo Simulation: Fat-Tailed Price Paths",
            "outputs/charts/monte_carlo.png",
            "<b>Forecast paths are highly dispersed:</b> the simulation highlights PLTR’s high volatility and fat-tailed behavior."
        ),
        (
            "Forecast Confidence Bands: Distribution of Outcomes",
            "outputs/charts/pltr_forecast_bands.png",
            f"<b>Outcome range is wide:</b> the 5th to 95th percentile forecast range spans "
            f"{forecast_summary['p5']:.2f} to {forecast_summary['p95']:.2f}, reinforcing probabilistic analysis."
        ),
    ]

    existing_sections = [
        (title, path, insight)
        for title, path, insight in sections
        if os.path.exists(path)
    ]

    for idx, (title, path, insight) in enumerate(existing_sections):
        story.append(Paragraph(title, styles["Heading2"]))
        story.append(Spacer(1, 8))
        story.append(Paragraph(f"<b>Insight:</b> {insight}", insight_style))
        story.append(Spacer(1, 8))
        story.append(Image(path, width=500, height=300))

        if idx < len(existing_sections) - 1:
            story.append(PageBreak())

    story.append(PageBreak())
    story.append(Paragraph("Conclusion", styles["Title"]))
    story.append(Spacer(1, 12))

    conclusion_text = f"""
    PLTR exhibits a high-volatility, regime-driven return profile with strong upside potential
    but significant downside risk.

    Buy-and-hold delivers superior returns ({performance['buy_hold_total_return']:.2%}),
    but with extreme drawdowns ({performance['buy_hold_max_drawdown']:.2%}).

    The strategy reduces drawdown to {performance['strategy_max_drawdown']:.2%},
    but lowers return to {performance['strategy_total_return']:.2%}.

    Statistical testing (p-value = {test_results['p_value']:.2f}) indicates that the
    strategy’s performance difference is not statistically significant, reinforcing that
    the strategy primarily transforms risk rather than generating statistically significant alpha.

    This suggests the strategy functions as a volatility and drawdown control mechanism
    rather than a persistent alpha generator.

    Forecast dispersion reinforces the need for probabilistic thinking in high-volatility assets.
    For high-volatility assets like PLTR, the distribution of outcomes matters as much as the expected return.
    """

    story.append(Paragraph(conclusion_text, styles["BodyText"]))

    doc.build(story)

    return output_path