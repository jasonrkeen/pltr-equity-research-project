from concurrent.futures import ThreadPoolExecutor, as_completed
from src.data_loader import load_data


def get_cache_path(ticker):
    clean_ticker = (
        ticker.replace("^", "")
        .replace("/", "_")
        .replace("-", "_")
        .lower()
    )

    return f"data/raw/{clean_ticker}.csv"


def load_asset(ticker, start, end, use_cache=True):
    cache_path = get_cache_path(ticker) if use_cache else None

    return load_data(
        ticker=ticker,
        start=start,
        end=end,
        cache_path=cache_path
    )


def load_assets(tickers, start, end, use_cache=True, max_workers=6):
    data = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(load_asset, ticker, start, end, use_cache): ticker
            for ticker in tickers
        }

        for future in as_completed(futures):
            ticker = futures[future]

            try:
                data[ticker] = future.result()
            except Exception as e:
                print(f"Failed to load {ticker}: {e}")
                data[ticker] = None

    return data