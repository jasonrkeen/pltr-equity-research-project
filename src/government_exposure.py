import pandas as pd
import os


def build_government_exposure_summary(df, ita):
    """
    Adds quantitative + qualitative government/defense exposure analysis
    """

    # =========================
    # Quantitative Metrics
    # =========================
    merged = df[["date", "buy_hold_return"]].merge(
        ita[["date", "daily_return"]],
        on="date",
        how="inner"
    ).dropna()

    merged.rename(columns={
        "buy_hold_return": "pltr_return",
        "daily_return": "ita_return"
    }, inplace=True)

    correlation = merged["pltr_return"].corr(merged["ita_return"])

    pltr_total_return = (1 + merged["pltr_return"]).prod() - 1
    ita_total_return = (1 + merged["ita_return"]).prod() - 1

    return_spread = pltr_total_return - ita_total_return

    # =========================
    # Summary Table
    # =========================
    data = {
        "category": [
            "Government revenue exposure",
            "Commercial revenue exposure",
            "Federal contract dependency",
            "Defense-sector correlation",
            "Return vs Defense Sector (ITA)"
        ],
        "value": [
            "High (based on company disclosures)",
            "Growing but secondary",
            "Material",
            f"{correlation:.2f}",
            f"{return_spread:.2%}"
        ],
        "interpretation": [
            "Revenue tied to public-sector budgets and contracts",
            "Commercial growth reduces dependency over time",
            "Contracts create revenue visibility but introduce concentration risk",
            "Moderate correlation indicates partial linkage to defense sector",
            "Relative performance vs ITA highlights PLTR’s hybrid tech/defense behavior"
        ]
    }

    summary = pd.DataFrame(data)

    os.makedirs("outputs/tables", exist_ok=True)
    summary.to_csv("outputs/tables/government_exposure_summary.csv", index=False)

    return summary