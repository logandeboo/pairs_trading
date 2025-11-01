import pandas as pd
from pathlib import Path
from datetime import datetime
import ast
from src.time_series_utils import filter_price_history_series_or_df_by_date_inclusive

_TICKER_COLUMN_NAME = "ticker"


def get_path_to_stock_price_history(ticker: str) -> Path:
    path_to_price_history_dir = Path("data/adj_close_price_data")
    return path_to_price_history_dir / f"{ticker}.csv"

def get_daily_price_history_df(ticker: str) -> pd.DataFrame:
    path_to_stock_price_history = get_path_to_stock_price_history(ticker)
    try:
        return pd.read_csv(path_to_stock_price_history, index_col=0, parse_dates=True)
    except FileNotFoundError:
        print(f"No price data found for ticker: {ticker}")
        return pd.DataFrame()

def get_all_tickers_price_history_df(
    start_date: datetime, end_date: datetime
) -> pd.DataFrame:
    all_tickers_price_history = []
    for ticker in get_ticker_list():
        path_to_ticker_price_history_csv = (
            Path("data") / "adj_close_price_data" / f"{ticker}.csv"
        )
        try:
            all_tickers_price_history.append(
                pd.read_csv(
                    path_to_ticker_price_history_csv, index_col=0, parse_dates=True
                )
            )
        except FileNotFoundError:
            print(f"No data found for ticker: {ticker}")
            continue
    all_tickers_price_history_df = pd.concat(all_tickers_price_history, axis=1)
    return filter_price_history_series_or_df_by_date_inclusive(
        start_date, end_date, all_tickers_price_history_df
    )


def get_benchmark_price_history_df(
    start_date: datetime, end_date: datetime, benchmark_ticker: str
) -> pd.DataFrame:
    path_to_benchmark_price_history_data = Path(
        f"data/adj_close_price_data_benchmarks/{benchmark_ticker}.csv"
    )
    benchmark_price_history_df = pd.read_csv(
        path_to_benchmark_price_history_data, index_col=0, parse_dates=True
    )
    return filter_price_history_series_or_df_by_date_inclusive(
        start_date, end_date, benchmark_price_history_df
    )


def get_tmp_hurst_exps_from_disk() -> pd.DataFrame:
    pairs_and_hurst_exponents = pd.read_csv(
        "scratchwork/hurst_exponents.csv", index_col=0
    )
    pairs_and_hurst_exponents["0"] = pairs_and_hurst_exponents["0"].apply(
        ast.literal_eval
    )
    pairs_and_hurst_exponents.columns = ["ticker_pair", "hurst_exp"]
    return pairs_and_hurst_exponents
