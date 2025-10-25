import pandas as pd
import numpy as np
from datetime import datetime
from typing import Collection
from src.pair_selection import (
    get_pair_price_history_df_filtered_by_date,
)
from src.statistical_utils import (
    get_pair_spread_series,
    create_returns_from_price_history,
    get_pair_spread_rolling_z_score_series,
)
from src.time_series_utils import subtract_n_us_trading_days_from_date

_TRADE_SIDE_COLUMN_NAME_SUFFIX = "_trade_side"
_ENTRANCE_THRESHOLD_IN_STANDARD_DEVIATIONS = 2
_EXIT_THRESHOLD_IN_STANDARD_DEVIATIONS = 0
_EXIT_THRESHOLD_ABSOLUTE_TOLERANCE = 0.1


# NOTE consider triggering condition as the spread converges back
# within trade threshold instead of upon exit.
def is_exit_condition_met(
    spread_z_score_series: pd.Series,
    t_minus_one_date: datetime,
    t_minus_two_date: datetime,
    exit_threshold: float,
    exit_threshold_proximity_tolerance: float,
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
    if np.isclose(
        z_score_at_t_minus_one_date,
        exit_threshold,
        atol=exit_threshold_proximity_tolerance,
    ):
        return True
    return False


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
    is_long_ticker_one: bool,
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


# TODO clean this up
def get_pair_trade_signals_by_ticker_df(
    ticker_one: str,
    ticker_two: str,
    z_scored_spread_series: pd.Series,
) -> pd.DataFrame:
    ticker_one_trade_side_column_name = ticker_one + _TRADE_SIDE_COLUMN_NAME_SUFFIX
    ticker_two_trade_side_column_name = ticker_two + _TRADE_SIDE_COLUMN_NAME_SUFFIX
    weight_at_date_df = pd.DataFrame(
        index=z_scored_spread_series.index,
        columns=[ticker_one_trade_side_column_name, ticker_two_trade_side_column_name],
    )
    is_invested = False
    for i, cur_date in enumerate(z_scored_spread_series.index):
        t_minus_one_date = z_scored_spread_series.index[i - 1]
        t_minus_two_date = z_scored_spread_series.index[i - 2]
        z_score_at_t_minus_one_date = z_scored_spread_series.loc[t_minus_one_date]
        if (
            z_score_at_t_minus_one_date >= _ENTRANCE_THRESHOLD_IN_STANDARD_DEVIATIONS
            and not is_invested
        ):
            weight_at_date_df = enter_position_at_current_date(
                weight_at_date_df,
                cur_date,
                ticker_one_trade_side_column_name,
                ticker_two_trade_side_column_name,
                is_long_ticker_one=True,
            )
            is_invested = True
        elif is_invested and is_exit_condition_met(
            z_scored_spread_series,
            t_minus_one_date,
            t_minus_two_date,
            _EXIT_THRESHOLD_IN_STANDARD_DEVIATIONS,
            _EXIT_THRESHOLD_ABSOLUTE_TOLERANCE,
        ):
            weight_at_date_df = set_weights_to_zero_at_current_date(
                weight_at_date_df,
                cur_date,
                ticker_one_trade_side_column_name,
                ticker_two_trade_side_column_name,
            )
            is_invested = False
        elif (
            z_score_at_t_minus_one_date <= -_ENTRANCE_THRESHOLD_IN_STANDARD_DEVIATIONS
            and not is_invested
        ):
            weight_at_date_df = enter_position_at_current_date(
                weight_at_date_df,
                cur_date,
                ticker_one_trade_side_column_name,
                ticker_two_trade_side_column_name,
                is_long_ticker_one=False,
            )
            is_invested = True
        else:
            weight_at_date_df = assign_weights_from_prev_date_to_cur_date(
                weight_at_date_df, cur_date, t_minus_one_date
            )
    return weight_at_date_df.fillna(0)


def get_single_pair_returns_by_ticker_df(
    ticker_one: str,
    ticker_two: str,
    trade_signals_df: pd.DataFrame,
    pair_price_history_df: pd.DataFrame,
) -> pd.DataFrame:
    trade_returns_for_tickers_df = pd.DataFrame()
    price_returns_df = create_returns_from_price_history(pair_price_history_df)
    ticker_one_trade_side_column_name = ticker_one + _TRADE_SIDE_COLUMN_NAME_SUFFIX
    ticker_two_trade_side_column_name = ticker_two + _TRADE_SIDE_COLUMN_NAME_SUFFIX
    signals_and_returns_df = trade_signals_df.merge(
        price_returns_df, left_index=True, right_index=True
    )
    trade_returns_for_tickers_df[ticker_one] = (
        signals_and_returns_df[ticker_one_trade_side_column_name]
        * signals_and_returns_df[ticker_one]
    )
    trade_returns_for_tickers_df[ticker_two] = (
        signals_and_returns_df[ticker_two_trade_side_column_name]
        * signals_and_returns_df[ticker_two]
    )
    return trade_returns_for_tickers_df


def get_all_pair_returns_by_ticker_df(
    simulation_start_date: datetime,
    simulation_end_date: datetime,
    pairs: Collection[tuple[str, str, float]],
    all_tickers_price_history_df: pd.DataFrame,
    *,
    spread_z_score_rolling_window_in_trading_days: int,
) -> pd.DataFrame:
    pair_returns_dfs = []
    start_date_for_simulation_adj_for_z_score_rolling_window = (
        subtract_n_us_trading_days_from_date(
            simulation_start_date,
            offset_in_us_trading_days=spread_z_score_rolling_window_in_trading_days,
        )
    )
    for (
        ticker_one,
        ticker_two,
    ) in pairs:
        pair_price_history_df = get_pair_price_history_df_filtered_by_date(
            ticker_one,
            ticker_two,
            start_date_for_simulation_adj_for_z_score_rolling_window,
            simulation_end_date,
            all_tickers_price_history_df,
        )
        spread_rolling_z_score_series = get_pair_spread_rolling_z_score_series(
            simulation_start_date,
            start_date_for_simulation_adj_for_z_score_rolling_window,
            simulation_end_date,
            pair_price_history_df,
            z_score_window_in_days=spread_z_score_rolling_window_in_trading_days,
        )
        pair_trade_signals_by_ticker_df = get_pair_trade_signals_by_ticker_df(
            ticker_one, ticker_two, spread_rolling_z_score_series
        )
        simulated_pair_returns_by_ticker_df = get_single_pair_returns_by_ticker_df(
            ticker_one,
            ticker_two,
            pair_trade_signals_by_ticker_df,
            pair_price_history_df,
        )
        pair_returns_dfs.append(simulated_pair_returns_by_ticker_df)
    pair_returns_by_ticker_df = pd.concat(
        pair_returns_dfs,
        axis=1,
    ).fillna(0)
    return pair_returns_by_ticker_df / 100


# if __name__ == "__main__":
# NOTE dates are the one year period following the two year period
# that was screened for cointegration
# z_score_window_in_calendar_days = 15
# num_pairs = 3
# start_date = datetime(2024, 1, 1)
# end_date = datetime(2024, 12, 31)
# pairs_and_hurst_exponents = get_tmp_hurst_exps_from_disk()
# pairs_and_hurst_exponents_top_n = pairs_and_hurst_exponents.sort_values(
#     by="hurst_exp"
# ).head(num_pairs)
# all_tickers_price_history_dict = get_all_tickers_price_history_df()
# trade_returns_for_all_tickers_df = pd.DataFrame()
# for ticker_one, ticker_two in pairs_and_hurst_exponents_top_n["ticker_pair"]:
#     pair_price_history_df = get_pair_price_history_df_algined_on_date(
#         ticker_one,
#         ticker_two,
#         start_date_adj_for_z_score_window,
#         end_date,
#         all_tickers_price_history_dict,
#     )
#     gamma = calculate_gamma(pair_price_history_df, start_date, end_date)
#     spread_series = calculate_spread(
#         pair_price_history_df, gamma, start_date_adj_for_z_score_window, end_date
#     )
#     z_scored_spread_series = calculate_rolling_zscore(
#         spread_series,
#         start_date,
#         end_date,
#         z_score_window_in_days=z_score_window_in_calendar_days,
#     )
#     trade_signals_df = create_trade_signals_from_z_scored_spread(
#         ticker_one,
#         ticker_two,
#         z_scored_spread_series,
#         _EXIT_THRESHOLD_ABSOLUTE_TOLERANCE,
#     )
#     price_returns_df = create_returns_from_price_history(pair_price_history_df)
#     trade_returns_for_tickers_in_pair_df = (
#         calculate_trade_returns_for_tickers_in_pair(
#             ticker_one, ticker_two, trade_signals_df, price_returns_df
#         )
#     )
#     trade_returns_for_all_tickers_df = pd.concat(
#         [trade_returns_for_all_tickers_df, trade_returns_for_tickers_in_pair_df],
#         axis=1,
#     )
# portfolio_performance_result = get_portfolio_performance_result(
#     trade_returns_for_all_tickers_df
# )
# print(portfolio_performance_result)
