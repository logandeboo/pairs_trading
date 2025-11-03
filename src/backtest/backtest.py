import pandas as pd
import numpy as np
from datetime import datetime
from typing import Collection, Sequence
from src.pair_selection import (
    filter_price_history_df_by_pair_and_date,
)
from src.statistical_utils import (
    create_daily_returns,
    get_pair_spread_rolling_z_score_series,
)
from src.time_series_utils import (
    subtract_n_us_trading_days_from_date,
    ONE_DAY_IN_TRADING_DAYS,
)
from src.backtest.backtest_config import BacktestConfig
from src.backtest.backtest_result import BacktestResult
from src.time_series_utils import (
    add_n_us_trading_days_to_date,
    subtract_n_us_trading_days_from_date
)
from src.pair_selection import get_pairs_to_backtest
from src.time_series_utils import get_rebalance_dates


_TRADE_SIDE_COLUMN_NAME_SUFFIX = "_trade_side"
_LONG_POSITION_FLAG = 1
_SHORT_POSITION_FLAG = -1

class Backtest:

    def __init__(self, backtest_config: BacktestConfig) -> None:
        self.backtest_config = backtest_config
    
        

    def run(self) -> BacktestResult:
        rebalance_dates = self.get_rebalance_dates(
            self.backtest_config.start_date,
            self.backtest_config.end_date,
            self.backtest_config.rebalance_freq_in_trading_days)
        for rebalance_date in rebalance_dates:
            pairs = get_pairs_to_backtest(self.backtest_config, rebalance_date)



# NOTE consider triggering condition as the spread converges back
# within trade threshold instead of upon exit.
def is_exit_condition_met(
    spread_z_score_series: pd.Series,
    t_minus_one_date: datetime,
    t_minus_two_date: datetime,
) -> bool:
    z_score_at_t_minus_one_date = spread_z_score_series.loc[t_minus_one_date]
    z_score_at_t_minus_two_date = spread_z_score_series.loc[t_minus_two_date]
    # upwards mean reversion across exit threshold
    if (
        z_score_at_t_minus_one_date > _EXIT_THRESHOLD_IN_STANDARD_DEVIATIONS
        and z_score_at_t_minus_two_date < _EXIT_THRESHOLD_IN_STANDARD_DEVIATIONS
    ):
        return True
    # downwards mean reversion across exit threshold
    if (
        z_score_at_t_minus_one_date < _EXIT_THRESHOLD_IN_STANDARD_DEVIATIONS
        and z_score_at_t_minus_two_date > _EXIT_THRESHOLD_IN_STANDARD_DEVIATIONS
    ):
        return True
    if np.isclose(
        z_score_at_t_minus_one_date,
        _EXIT_THRESHOLD_IN_STANDARD_DEVIATIONS,
        atol=_EXIT_THRESHOLD_ABSOLUTE_TOLERANCE,
    ):
        return True
    return False


def set_trade_signals_to_zero_for_date(
    pair_trade_signals_by_date_df: pd.DataFrame,
    signal_date: datetime,
) -> pd.DataFrame:
    pair_trade_signals_by_date_df.loc[signal_date] = [0, 0]


def enter_position_at_current_date(
    pair_trade_signals_by_date_df: pd.DataFrame,
    date: datetime,
    ticker_one_trade_side_column_name: str,
    ticker_two_trade_side_column_name: str,
    *,
    is_long_ticker_one: bool,
) -> pd.DataFrame:

    if is_long_ticker_one:
        pair_trade_signals_by_date_df.loc[date, ticker_one_trade_side_column_name] = (
            _LONG_POSITION_FLAG
        )
        pair_trade_signals_by_date_df.loc[date, ticker_two_trade_side_column_name] = (
            _SHORT_POSITION_FLAG
        )
    else:
        pair_trade_signals_by_date_df.loc[date, ticker_one_trade_side_column_name] = (
            _SHORT_POSITION_FLAG
        )
        pair_trade_signals_by_date_df.loc[date, ticker_two_trade_side_column_name] = (
            _LONG_POSITION_FLAG
        )
    return pair_trade_signals_by_date_df


def set_short_spread_trade_positions_for_date(
    pair_trade_signals_by_date_df: pd.DataFrame,
    signal_date: datetime,
) -> pd.DataFrame:
    pair_trade_signals_by_date_df.loc[signal_date] = [
        _LONG_POSITION_FLAG,
        _SHORT_POSITION_FLAG,
    ]


def set_long_spread_trade_positions_for_date(
    pair_trade_signals_by_date_df: pd.DataFrame,
    signal_date: datetime,
) -> pd.DataFrame:
    pair_trade_signals_by_date_df.loc[signal_date] = [
        _SHORT_POSITION_FLAG,
        _LONG_POSITION_FLAG,
    ]


def set_positions_from_prev_date_to_cur_date(
    pair_trade_signals_by_date_df: pd.DataFrame,
    cur_date: datetime,
    t_minus_one_date: datetime,
) -> pd.DataFrame:
    pair_trade_signals_by_date_df.loc[cur_date] = pair_trade_signals_by_date_df.loc[
        t_minus_one_date
    ]


def should_enter_short_spread_trade(
    z_score_at_date_t_minus_one: float, *, is_invested: bool
) -> bool:
    return (
        z_score_at_date_t_minus_one >= _ENTRANCE_THRESHOLD_IN_STANDARD_DEVIATIONS
        and not is_invested
    )


def should_enter_long_spread_trade(
    z_score_at_date_t_minus_one: float, *, is_invested: bool
) -> bool:
    return (
        z_score_at_date_t_minus_one <= -1 * _ENTRANCE_THRESHOLD_IN_STANDARD_DEVIATIONS
        and not is_invested
    )


def init_pair_trade_positions_by_date_df(
    ticker_one: str,
    ticker_two: str,
    z_scored_spread_series: pd.Series,
) -> pd.DataFrame:
    ticker_one_trade_side_column_name = ticker_one + _TRADE_SIDE_COLUMN_NAME_SUFFIX
    ticker_two_trade_side_column_name = ticker_two + _TRADE_SIDE_COLUMN_NAME_SUFFIX
    return pd.DataFrame(
        index=z_scored_spread_series.index,
        columns=[ticker_one_trade_side_column_name, ticker_two_trade_side_column_name],
    )


def should_exit_current_trade(
    spread_rolling_z_score_series: pd.Series,
    t_minus_one_date: datetime,
    t_minus_two_date: datetime,
    *,
    is_invested: bool,
) -> bool:
    if not is_invested:
        return False
    z_score_at_t_minus_one_date = spread_rolling_z_score_series.loc[t_minus_one_date]
    z_score_at_t_minus_two_date = spread_rolling_z_score_series.loc[t_minus_two_date]
    # upwards mean reversion across exit threshold
    if (
        z_score_at_t_minus_one_date > _EXIT_THRESHOLD_IN_STANDARD_DEVIATIONS
        and z_score_at_t_minus_two_date < _EXIT_THRESHOLD_IN_STANDARD_DEVIATIONS
    ):
        return True
    # downwards mean reversion across exit threshold
    if (
        z_score_at_t_minus_one_date < _EXIT_THRESHOLD_IN_STANDARD_DEVIATIONS
        and z_score_at_t_minus_two_date > _EXIT_THRESHOLD_IN_STANDARD_DEVIATIONS
    ):
        return True
    if np.isclose(
        z_score_at_t_minus_one_date,
        _EXIT_THRESHOLD_IN_STANDARD_DEVIATIONS,
        atol=_EXIT_THRESHOLD_ABSOLUTE_TOLERANCE,
    ):
        return True
    return False


# NOTE implementation was changed in caller to add an extra day to
# the beginning of the z score series so a trade can be initiated
# on day 0 of the backtest period. Hence why the loop starts at 1.
# Day 0 is one day before the backtest period start.
# TODO this should be refactored
def get_daily_pair_trade_signals_df(
    ticker_one: str,
    ticker_two: str,
    spread_rolling_z_score_series: pd.Series,
) -> pd.DataFrame:
    pair_trade_positions_by_date_df = init_pair_trade_positions_by_date_df(
        ticker_one, ticker_two, spread_rolling_z_score_series
    )
    is_invested = False
    for i in range(1, len(spread_rolling_z_score_series)):
        cur_date = spread_rolling_z_score_series.index[i]
        t_minus_one_date = spread_rolling_z_score_series.index[i - 1]
        z_score_at_t_minus_one_date = spread_rolling_z_score_series.loc[
            t_minus_one_date
        ]
        if should_enter_short_spread_trade(
            z_score_at_t_minus_one_date, is_invested=is_invested
        ):
            set_short_spread_trade_positions_for_date(
                pair_trade_positions_by_date_df,
                cur_date,
            )
            is_invested = True
        elif should_exit_current_trade(
            spread_rolling_z_score_series,
            t_minus_one_date,
            spread_rolling_z_score_series.index[i - 2],
            is_invested=is_invested,
        ):
            set_trade_signals_to_zero_for_date(
                pair_trade_positions_by_date_df,
                cur_date,
            )
            is_invested = False
        elif should_enter_long_spread_trade(
            z_score_at_t_minus_one_date, is_invested=is_invested
        ):
            set_long_spread_trade_positions_for_date(
                pair_trade_positions_by_date_df,
                cur_date,
            )
            is_invested = True
        else:
            set_positions_from_prev_date_to_cur_date(
                pair_trade_positions_by_date_df, cur_date, t_minus_one_date
            )
    return pair_trade_positions_by_date_df.fillna(0)


def convert_percent_returns_to_decimal(df: pd.DataFrame) -> pd.DataFrame:
    return df / 100


def get_paired_tickers_daily_return_df(
    ticker_one: str,
    ticker_two: str,
    trade_signals_df: pd.DataFrame,
    pair_price_history_df: pd.DataFrame,
) -> pd.DataFrame:
    trade_returns_for_tickers_df = pd.DataFrame()
    price_returns_df = create_daily_returns(pair_price_history_df)
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
    return convert_percent_returns_to_decimal(trade_returns_for_tickers_df)


# TODO this needs to be cleaned up
def get_backtest_returns_by_ticker_df(
    backtest_start_date: datetime,
    backtest_end_date: datetime,
    pairs: Collection[tuple[str, str]],
    all_tickers_price_history_df: pd.DataFrame,
    *,
    spread_z_score_rolling_window_in_trading_days: int,
) -> pd.DataFrame:
    all_tickers_daily_return_dfs = []
    backtest_start_date_adj_for_z_score_rolling_window = (
        subtract_n_us_trading_days_from_date(
            backtest_start_date,
            offset_in_us_trading_days=spread_z_score_rolling_window_in_trading_days
            + ONE_DAY_IN_TRADING_DAYS,
        )
    )
    for ticker_one, ticker_two in pairs:
        pair_price_history_df = filter_price_history_df_by_pair_and_date(
            ticker_one,
            ticker_two,
            backtest_start_date_adj_for_z_score_rolling_window,
            backtest_end_date,
            all_tickers_price_history_df,
        )
        spread_rolling_z_score_series = get_pair_spread_rolling_z_score_series(
            backtest_start_date_adj_for_z_score_rolling_window,
            backtest_end_date,
            pair_price_history_df,
            z_score_window_in_trading_days=spread_z_score_rolling_window_in_trading_days,
        )
        breakpoint()
        daily_pair_trade_signals_df = get_daily_pair_trade_signals_df(
            ticker_one, ticker_two, spread_rolling_z_score_series
        )
        all_tickers_daily_return_dfs.append(
            get_paired_tickers_daily_return_df(
                ticker_one,
                ticker_two,
                daily_pair_trade_signals_df,
                pair_price_history_df,
            )
        )
    return pd.concat(
        all_tickers_daily_return_dfs,
        axis=1,
    ).fillna(0)


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
