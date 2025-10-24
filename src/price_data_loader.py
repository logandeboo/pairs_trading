import pandas as pd
from pathlib import Path
from datetime import datetime
from src.time_series_utils import (
    filter_price_history_series_or_df_by_date
)

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