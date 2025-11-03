import pandas as pd
from datetime import datetime
from typing import Union, Sequence
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


def filter_price_history_series_or_df_by_date_inclusive(
    start_date: datetime,
    end_date: datetime,
    df_or_series_with_date_index: Union[pd.DataFrame, pd.Series],
) -> Union[pd.DataFrame, pd.Series]:
    return df_or_series_with_date_index[
        (df_or_series_with_date_index.index >= start_date)
        & (df_or_series_with_date_index.index <= end_date)
    ]


def filter_price_history_df_by_pair_and_date(
    ticker_one: str,
    ticker_two: str,
    start_date: datetime,
    end_date: datetime,
    all_tickers_price_history_df: pd.DataFrame,
) -> pd.DataFrame:
    pair_price_history_df = all_tickers_price_history_df[[ticker_one, ticker_two]]
    return filter_price_history_series_or_df_by_date_inclusive(
        start_date, end_date, pair_price_history_df
    ).dropna()

def get_rebalance_dates(
    start_date: datetime,
    end_date: datetime,
    rebalance_freq_in_trading_days: int,
) -> Sequence[datetime]:
    all_rebalance_dates = []
    date = start_date
    while date < end_date:
        all_rebalance_dates.append(date)
        date = add_n_us_trading_days_to_date(
            date,
            rebalance_freq_in_trading_days
        )
    return all_rebalance_dates
