import pandas as pd
from pathlib import Path
from datetime import datetime
import pickle
from typing import Mapping
import ast
from src.time_series_utils import filter_price_history_series_or_df_by_date

_TICKER_COLUMN_NAME = "ticker"


def get_ticker_list() -> list[tuple[str, str]]:
    path_to_ticker_list = Path("data/russell_3000_constituents.csv")
    ticker_df = pd.read_csv(path_to_ticker_list)
    return ticker_df[_TICKER_COLUMN_NAME].to_list()


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
    return filter_price_history_series_or_df_by_date(
        start_date, end_date, all_tickers_price_history_df
    )


def create_ticker_to_sector_map() -> Mapping[str, str]:
    with open("data/ticker_to_sector.pkl", "rb") as ticker_to_sector_map_file:
        return pickle.load(ticker_to_sector_map_file)


def get_benchmark_price_history_df(
    start_date: datetime, end_date: datetime, benchmark_ticker: str
) -> pd.DataFrame:
    path_to_benchmark_price_history_data = Path(
        f"data/adj_close_price_data_benchmarks/{benchmark_ticker}.csv"
    )
    benchmark_price_history_df = pd.read_csv(
        path_to_benchmark_price_history_data, index_col=0, parse_dates=True
    )
    return filter_price_history_series_or_df_by_date(
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
