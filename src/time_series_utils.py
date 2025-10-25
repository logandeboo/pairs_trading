import pandas as pd
from datetime import datetime
from typing import Mapping, Union
from pandas.tseries.holiday import USFederalHolidayCalendar, GoodFriday
from pandas.tseries.offsets import CustomBusinessDay

ONE_YEAR_IN_TRADING_DAYS = 252
ONE_DAY_IN_TRADING_DAYS = 1
_COLUMBUS_DAY_HOLIDAY_STRING = "Columbus Day"
_VETRANS_DAY_HOLIDAY_STRING = "Veterans Day"


class USTradingCalendar(USFederalHolidayCalendar):
    rules = [
        r
        for r in USFederalHolidayCalendar.rules
        if r.name not in [_COLUMBUS_DAY_HOLIDAY_STRING, _VETRANS_DAY_HOLIDAY_STRING]
    ] + [GoodFriday]


def subtract_n_us_trading_days_from_date(
    date: datetime, *, offset_in_us_trading_days: int
) -> datetime:
    trading_day = CustomBusinessDay(calendar=USTradingCalendar())
    return (
        pd.Timestamp(date) - offset_in_us_trading_days * trading_day
    ).to_pydatetime()


def add_n_us_trading_days_to_date(
    date: datetime, offset_in_us_trading_days: int
) -> datetime:
    trading_day = CustomBusinessDay(calendar=USTradingCalendar())
    return (
        pd.Timestamp(date) + offset_in_us_trading_days * trading_day
    ).to_pydatetime()


def filter_price_history_series_or_df_by_date(
    start_date: datetime,
    end_date: datetime,
    stock_and_benchmark_price_history_df: Union[pd.DataFrame, pd.Series],
) -> Union[pd.DataFrame, pd.Series]:
    return stock_and_benchmark_price_history_df[
        (stock_and_benchmark_price_history_df.index >= start_date)
        & (stock_and_benchmark_price_history_df.index <= end_date)
    ]


def get_pair_price_history_df_filtered_by_date(
    ticker_one: str,
    ticker_two: str,
    start_date: datetime,
    end_date: datetime,
    all_tickers_price_history_df: pd.DataFrame,
) -> pd.DataFrame:
    pair_price_history_df = all_tickers_price_history_df[[ticker_one, ticker_two]]
    return filter_price_history_series_or_df_by_date(
        start_date, end_date, pair_price_history_df
    ).dropna()
