import os
import pandas as pd
import yfinance as yf


def load_data(ticker, start, end, cache_path=None):
    if cache_path and os.path.exists(cache_path):
        return pd.read_csv(cache_path, parse_dates=["date"])

    df = yf.download(ticker, start=start, end=end, progress=False)

    if df.empty:
        raise ValueError(f"No data downloaded for {ticker}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    df.columns = [col.lower().replace(" ", "_") for col in df.columns]

    if "adj_close" in df.columns:
        df["close"] = df["adj_close"]

    df = df.sort_values("date").drop_duplicates(subset=["date"])

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df.to_csv(cache_path, index=False)

    return df


def save_raw_data(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)