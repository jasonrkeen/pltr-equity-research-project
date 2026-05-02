import os
import requests
import pandas as pd


PLTR_CIK = "0001321655"


def fetch_company_facts(cik=PLTR_CIK):
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

    headers = {
        "User-Agent": "Jason Keen keenjasonr@gmail.com"
    }

    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()

    return response.json()


def extract_usd_fact(facts_json, tag):
    try:
        facts = facts_json["facts"]["us-gaap"][tag]["units"]["USD"]
    except KeyError:
        return pd.DataFrame()

    df = pd.DataFrame(facts)

    if df.empty:
        return df

    df = df[["fy", "fp", "form", "filed", "end", "val"]].copy()
    df = df[df["form"].isin(["10-K", "10-Q"])]
    df = df.drop_duplicates(subset=["fy", "fp", "form", "end"], keep="last")

    df["end"] = pd.to_datetime(df["end"])
    df["val"] = pd.to_numeric(df["val"], errors="coerce")

    return df.sort_values("end")


def safe_merge(left, right, col_name):
    if right.empty:
        left[col_name] = pd.NA
        return left

    return left.merge(
        right.rename(columns={"val": col_name})[
            ["fy", "fp", "form", "end", col_name]
        ],
        on=["fy", "fp", "form", "end"],
        how="left"
    )


def add_ttm_metrics(summary):
    summary = summary.sort_values("end").copy()

    quarterly = summary[summary["fp"].isin(["Q1", "Q2", "Q3", "Q4"])].copy()

    quarterly["ttm_revenue"] = quarterly["revenue"].rolling(4).sum()
    quarterly["ttm_net_income"] = quarterly["net_income"].rolling(4).sum()
    quarterly["ttm_operating_cash_flow"] = quarterly["operating_cash_flow"].rolling(4).sum()
    quarterly["ttm_free_cash_flow"] = quarterly["free_cash_flow"].rolling(4).sum()

    quarterly["qoq_revenue_growth"] = quarterly["revenue"].pct_change()
    quarterly["ttm_revenue_growth"] = quarterly["ttm_revenue"].pct_change(4)
    quarterly["revenue_acceleration"] = quarterly["qoq_revenue_growth"].diff()

    quarterly["ttm_net_margin"] = quarterly["ttm_net_income"] / quarterly["ttm_revenue"]
    quarterly["ttm_fcf_margin"] = quarterly["ttm_free_cash_flow"] / quarterly["ttm_revenue"]

    return summary.merge(
        quarterly[
            [
                "fy",
                "fp",
                "form",
                "end",
                "ttm_revenue",
                "ttm_net_income",
                "ttm_operating_cash_flow",
                "ttm_free_cash_flow",
                "qoq_revenue_growth",
                "ttm_revenue_growth",
                "revenue_acceleration",
                "ttm_net_margin",
                "ttm_fcf_margin",
            ]
        ],
        on=["fy", "fp", "form", "end"],
        how="left"
    )


def build_fundamental_summary():
    facts = fetch_company_facts()

    revenue = extract_usd_fact(
        facts,
        "RevenueFromContractWithCustomerExcludingAssessedTax"
    )

    net_income = extract_usd_fact(facts, "NetIncomeLoss")
    operating_cf = extract_usd_fact(
        facts,
        "NetCashProvidedByUsedInOperatingActivities"
    )
    capex = extract_usd_fact(
        facts,
        "PaymentsToAcquirePropertyPlantAndEquipment"
    )

    if revenue.empty:
        raise ValueError("Revenue data not available")

    summary = revenue.rename(columns={"val": "revenue"})[
        ["fy", "fp", "form", "filed", "end", "revenue"]
    ]

    summary = safe_merge(summary, net_income, "net_income")
    summary = safe_merge(summary, operating_cf, "operating_cash_flow")
    summary = safe_merge(summary, capex, "capex")

    numeric_cols = [
        "revenue",
        "net_income",
        "operating_cash_flow",
        "capex"
    ]

    for col in numeric_cols:
        summary[col] = pd.to_numeric(summary[col], errors="coerce")

    summary["capex"] = summary["capex"].abs()
    summary["free_cash_flow"] = (
        summary["operating_cash_flow"] - summary["capex"]
    )

    annual = summary[summary["fp"] == "FY"].copy()
    annual["annual_revenue_growth"] = annual["revenue"].pct_change()

    summary = summary.merge(
        annual[["fy", "annual_revenue_growth"]],
        on="fy",
        how="left"
    )

    summary["net_margin"] = summary["net_income"] / summary["revenue"]
    summary["fcf_margin"] = summary["free_cash_flow"] / summary["revenue"]

    summary = add_ttm_metrics(summary)

    os.makedirs("outputs/tables", exist_ok=True)
    summary.to_csv("outputs/tables/fundamental_summary.csv", index=False)

    return summary