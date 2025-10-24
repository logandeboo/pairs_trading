import pandas as pd
from datetime import datetime
from typing import Mapping, Union

def filter_price_history_series_or_df_by_date(
    start_date: datetime,
    end_date: datetime,
    stock_and_benchmark_price_history_df: Union[pd.DataFrame, pd.Series],
) -> Union[pd.DataFrame, pd.Series]:
    return stock_and_benchmark_price_history_df[
        (stock_and_benchmark_price_history_df.index >= start_date)
        & (stock_and_benchmark_price_history_df.index <= end_date)
    ]

def get_pair_price_history_df_algined_on_date(
    ticker_one: str,
    ticker_two: str,
    start_date: datetime,
    end_date: datetime,
    all_tickers_price_history_dict: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    ticker_one_price_history_series = filter_price_history_series_or_df_by_date(
        start_date, end_date, all_tickers_price_history_dict[ticker_one]
    )
    ticker_two_price_history_series = filter_price_history_series_or_df_by_date(
        start_date, end_date, all_tickers_price_history_dict[ticker_two]
    )
    return pd.concat(
        [
            ticker_one_price_history_series,
            ticker_two_price_history_series,
        ],
        axis=1,
    ).dropna()