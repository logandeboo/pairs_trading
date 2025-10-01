import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import os
import statsmodels.api as sm
from collections.abc import Mapping
from pathlib import Path
from itertools import combinations
import ast
from typing import Sequence


def is_exit_condition_met(
    spread_z_score_series: pd.Series,
    t_minus_one_date: datetime,
    t_minus_two_date: datetime,
    exit_threshold: float,
) -> bool:
    z_score_at_t_minus_one_date = spread_z_score_series.loc[t_minus_one_date]
    z_score_at_t_minus_two_date = spread_z_score_series.loc[t_minus_two_date]
    # upwards mean reversion across exit threshold
    if (
        z_score_at_t_minus_one_date > exit_threshold
        and z_score_at_t_minus_two_date < exit_threshold
    ):
        return True
    # downwards mean reversion across exit threshold
    if (
        z_score_at_t_minus_one_date < exit_threshold
        and z_score_at_t_minus_two_date > exit_threshold
    ):
        return True
    if np.isclose(z_score_at_t_minus_one_date, exit_threshold, atol=0.1):
        return True
    return False


def get_date_of_previous_day(
    spread_z_score_series: pd.Series,
    index: int,
) -> datetime:
    return spread_z_score_series.index[index - 1]


def set_weights_to_zero_at_current_date(
    weight_at_date_df: pd.DataFrame,
    date: datetime,
    ticker_one_weight_column_name: str,
    ticker_two_weight_column_name: str,
) -> pd.DataFrame:
    weight_at_date_df.loc[date, ticker_one_weight_column_name] = 0
    weight_at_date_df.loc[date, ticker_two_weight_column_name] = 0
    return weight_at_date_df


def enter_position_at_current_date(
    weight_at_date_df: pd.DataFrame,
    date: datetime,
    ticker_one_weight_column_name: str,
    ticker_two_weight_column_name: str,
    *,
    is_long_ticker_one: bool
) -> pd.DataFrame:
    long_position_flag = 1
    short_position_flag = -1
    if is_long_ticker_one:
        weight_at_date_df.loc[date, ticker_one_weight_column_name] = long_position_flag
        weight_at_date_df.loc[date, ticker_two_weight_column_name] = short_position_flag
    else:
        weight_at_date_df.loc[date, ticker_one_weight_column_name] = short_position_flag
        weight_at_date_df.loc[date, ticker_two_weight_column_name] = long_position_flag
    return weight_at_date_df


def assign_weights_from_prev_date_to_cur_date(
    weight_at_date_df: pd.DataFrame,
    cur_date: datetime,
    t_minus_one_date: datetime,
) -> pd.DataFrame:
    weight_at_date_df.loc[cur_date] = weight_at_date_df.loc[t_minus_one_date]
    return weight_at_date_df


def create_trade_signals_for_spread_z_score_series(
    ticker_pair: tuple[str, str],
    spread_z_score_series: pd.Series,
) -> pd.DataFrame:
    ticker_one = ticker_pair[0]
    ticker_two = ticker_pair[1]
    weight_column_name_suffix = "_trade_side"
    ticker_one_weight_column_name = ticker_one + weight_column_name_suffix
    ticker_two_weight_column_name = ticker_two + weight_column_name_suffix

    weight_at_date_df = pd.DataFrame(
        index=spread_z_score_series.index,
        columns=[ticker_one_weight_column_name, ticker_two_weight_column_name],
    )
    entrance_threshold = 2
    exit_threshold = 0

    is_invested = False
    for i, (cur_date, z_score) in enumerate(spread_z_score_series.items()):
        # TODO there is a better way than to ignore the first two days.
        # probably just extend the spread_z_score_series
        if i < 2:
            continue
        t_minus_one_date = spread_z_score_series.index[i - 1]
        t_minus_two_date = spread_z_score_series.index[i - 2]
        z_score_at_t_minus_one_date = spread_z_score_series.loc[t_minus_one_date]

        if z_score_at_t_minus_one_date >= entrance_threshold and not is_invested:
            weight_at_date_df = enter_position_at_current_date(
                weight_at_date_df,
                cur_date,
                ticker_one_weight_column_name,
                ticker_two_weight_column_name,
                is_long_ticker_one=True,
            )
            is_invested = True

        elif is_invested and is_exit_condition_met(
            spread_z_score_series, t_minus_one_date, t_minus_two_date, exit_threshold
        ):
            weight_at_date_df = set_weights_to_zero_at_current_date(
                weight_at_date_df,
                cur_date,
                ticker_one_weight_column_name,
                ticker_two_weight_column_name,
            )
            is_invested = False

        elif z_score_at_t_minus_one_date <= -entrance_threshold and not is_invested:
            weight_at_date_df = enter_position_at_current_date(
                weight_at_date_df,
                cur_date,
                ticker_one_weight_column_name,
                ticker_two_weight_column_name,
                is_long_ticker_one=False,
            )
            is_invested = True
        else:
            weight_at_date_df = assign_weights_from_prev_date_to_cur_date(
                weight_at_date_df, cur_date, t_minus_one_date
            )

    return weight_at_date_df.fillna(0)


def calculate_rolling_zscore(
    spread: pd.Series,
    z_score_window_in_days: int,
    start_date: datetime,
    end_date: datetime,
) -> pd.Series:
    rolling_mean = spread.rolling(window=z_score_window_in_days).mean()
    rolling_std = spread.rolling(window=z_score_window_in_days).std()
    zscore_series = (spread - rolling_mean) / rolling_std
    zscore_filtered_series = zscore_series.loc[start_date:end_date]
    return zscore_filtered_series


def get_tmp_hurst_exps_from_disk() -> pd.DataFrame:
    pairs_and_hurst_exponents = pd.read_csv(
        "scratchwork/hurst_exponents.csv", index_col=0
    )
    pairs_and_hurst_exponents["0"] = pairs_and_hurst_exponents["0"].apply(
        ast.literal_eval
    )
    pairs_and_hurst_exponents[["ticker_one", "ticker_two"]] = pd.DataFrame(
        pairs_and_hurst_exponents["0"].tolist(), index=pairs_and_hurst_exponents.index
    )
    pairs_and_hurst_exponents = pairs_and_hurst_exponents.rename(
        columns={"1": "hurst_exp"}
    )
    return pairs_and_hurst_exponents[["ticker_one", "ticker_two", "hurst_exp"]]


if __name__ == "__main__":
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2023, 12, 31)
    tickers_and_hurst_exponents_df = get_tmp_hurst_exps_from_disk()

    breakpoint()
