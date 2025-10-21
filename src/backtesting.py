import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import os
import statsmodels.api as sm
from collections.abc import Mapping
from pathlib import Path
import ast
import sys
from typing import NamedTuple, Mapping

# TODO this is a temporary workaround
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# TODO add new file to hold common funcs
from src.pair_selection import (
    read_stock_price_history_into_dict,
    calculate_spread,
    calculate_historical_gamma,
    get_pair_price_history_df_algined_on_date,
    create_returns_from_price_history,
)


# TODO implement calculations for commented out class attributes
class PortfolioPerformanceResult(NamedTuple):
    annualized_return: float
    annualized_volatility: float
    annualized_sharpe_ratio: float
    # hit_rate: float
    # max_drawdown: float
    # max_drawdown_duration_in_trading_days: float
    # skewness: float
    performance_period_start: datetime
    performance_period_end: datetime


_TRADE_SIDE_COLUMN_NAME_SUFFIX = "_trade_side"
_ONE_WAY_T_COST_IN_BASIS_POINTS = 10
PORTFOLIO_DAILY_RETURN_COLUMN_NAME = "portfolio_daily_return"
_PORTFOLIO_GROSS_DAILY_RETURN_AFTER_T_COST_COLUMN_NAME = (
    "portfolio_gross_daily_return_after_t_cost"
)
_PORTFOLIO_DAILY_T_COST_COLUMN_NAME = "portfolio_daily_t_cost"
DAILY_TRANSACTION_COUNT_COLUMN_NAME = "transaction_count"
_EXIT_THRESHOLD_ABSOLUTE_TOLERANCE = 0.1
_ANNUALIZATION_FACTOR_IN_TRADING_DAYS = 252
_RISK_FREE_RATE_IN_DECIMAL = 0.04
_BASIS_POINTS_TO_DECIMAL_CONVERSION_FACTOR = 10_000
PORTFOLIO_DAILY_RETURN_AFTER_T_COST_COLUMN_NAME = "portfolio_daily_return_after_t_cost"


# TODO consider triggering condition as the spread converges back
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
def create_trade_signals_from_z_scored_spread(
    ticker_one: str,
    ticker_two: str,
    z_scored_spread_series: pd.Series,
    exit_threshold_proximity_tolerance: float,
) -> pd.DataFrame:
    ticker_one_trade_side_column_name = ticker_one + _TRADE_SIDE_COLUMN_NAME_SUFFIX
    ticker_two_trade_side_column_name = ticker_two + _TRADE_SIDE_COLUMN_NAME_SUFFIX

    weight_at_date_df = pd.DataFrame(
        index=z_scored_spread_series.index,
        columns=[ticker_one_trade_side_column_name, ticker_two_trade_side_column_name],
    )
    entrance_threshold = 2
    exit_threshold = 0

    is_invested = False
    for i, cur_date in enumerate(z_scored_spread_series.index):
        t_minus_one_date = z_scored_spread_series.index[i - 1]
        t_minus_two_date = z_scored_spread_series.index[i - 2]
        z_score_at_t_minus_one_date = z_scored_spread_series.loc[t_minus_one_date]
        if z_score_at_t_minus_one_date >= entrance_threshold and not is_invested:
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
            exit_threshold,
            exit_threshold_proximity_tolerance,
        ):
            weight_at_date_df = set_weights_to_zero_at_current_date(
                weight_at_date_df,
                cur_date,
                ticker_one_trade_side_column_name,
                ticker_two_trade_side_column_name,
            )
            is_invested = False
        elif z_score_at_t_minus_one_date <= -entrance_threshold and not is_invested:
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


def calculate_rolling_zscore(
    spread: pd.Series,
    start_date: datetime,
    end_date: datetime,
    *,
    z_score_window_in_days: int,
) -> pd.Series:
    rolling_mean = spread.rolling(window=z_score_window_in_days).mean()
    rolling_std = spread.rolling(window=z_score_window_in_days).std()
    rolling_z_score = (spread - rolling_mean) / rolling_std
    return rolling_z_score[
        (rolling_z_score.index >= start_date) & (rolling_z_score.index <= end_date)
    ]


def calculate_z_score_from_cointegrated_period(
    spread: pd.Series, start_date: datetime, end_date: datetime
) -> pd.Series:
    mean = spread.mean()
    std = spread.std()
    z_scored_spread = (spread - mean) / std
    return z_scored_spread[
        (z_scored_spread.index >= start_date) & (z_scored_spread.index <= end_date)
    ]


def get_tmp_hurst_exps_from_disk() -> pd.DataFrame:
    pairs_and_hurst_exponents = pd.read_csv(
        "scratchwork/hurst_exponents.csv", index_col=0
    )
    pairs_and_hurst_exponents["0"] = pairs_and_hurst_exponents["0"].apply(
        ast.literal_eval
    )
    pairs_and_hurst_exponents.columns = ["ticker_pair", "hurst_exp"]
    return pairs_and_hurst_exponents


def calculate_trade_returns_for_tickers_in_pair(
    ticker_one: str,
    ticker_two: str,
    trade_signals_df: pd.DataFrame,
    price_returns_df: pd.DataFrame,
) -> pd.DataFrame:
    trade_returns_for_tickers_df = pd.DataFrame()
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


def calculate_number_of_trades_per_day(
    trade_returns_for_all_tickers_df: pd.DataFrame,
) -> pd.DataFrame:
    trade_returns_t_minus_one_for_all_tickers_df = (
        trade_returns_for_all_tickers_df.shift(1).fillna(0)
    )
    buys = (trade_returns_for_all_tickers_df != 0) & (
        trade_returns_t_minus_one_for_all_tickers_df == 0
    )
    sells = (trade_returns_t_minus_one_for_all_tickers_df != 0) & (
        trade_returns_for_all_tickers_df == 0
    )
    # Sells are shifted up on day so t cost can be subtracted
    # from the last day the position was active
    transaction_count_series = buys.sum(axis=1) + sells.shift(-1).sum(axis=1)
    # Assumes all open trades are closed on last day of trading period
    num_trades_closed_on_last_day_of_period = (
        trade_returns_for_all_tickers_df.iloc[-1] != 0
    ).sum()
    transaction_count_series.iloc[-1] += num_trades_closed_on_last_day_of_period
    return transaction_count_series.astype("int64").to_frame(name="transaction_count")


def calculate_daily_portfolio_return_before_t_costs(
    trade_returns_for_all_tickers_df: pd.DataFrame,
) -> pd.DataFrame:
    daily_portfolio_return_df = pd.DataFrame()
    daily_portfolio_return_df[PORTFOLIO_DAILY_RETURN_COLUMN_NAME] = (
        trade_returns_for_all_tickers_df.sum(axis=1)
        / trade_returns_for_all_tickers_df.astype(bool).sum(axis=1)
    )
    return daily_portfolio_return_df


# Calculates total portfolio return on employed capital assuming equal dollar weight trades.
# Σ(active trade returns) / number of active trades
# For each day, the portfolio return is calculated as the sum of returns on open trades divided
# by the number of open positions. E.g., on day t, if there are two open trades with returns of
# .05% and 1.2%, the portfolio return is (0.0005 + 0.012) / 2
def calculate_daily_return_on_employed_capital_after_t_costs(
    trade_returns_for_all_tickers_df: pd.DataFrame,
    one_way_t_cost_in_basis_points: int,
) -> pd.DataFrame:
    daily_transaction_count_df = calculate_number_of_trades_per_day(
        trade_returns_for_all_tickers_df
    )
    daily_portfolio_return_df = calculate_daily_portfolio_return_before_t_costs(
        trade_returns_for_all_tickers_df
    )
    daily_portfolio_return_and_transaction_count_df = daily_portfolio_return_df.merge(
        daily_transaction_count_df,
        left_index=True,
        right_index=True,
    )
    daily_portfolio_return_and_transaction_count_df[
        _PORTFOLIO_DAILY_T_COST_COLUMN_NAME
    ] = daily_portfolio_return_and_transaction_count_df[
        DAILY_TRANSACTION_COUNT_COLUMN_NAME
    ] * (
        one_way_t_cost_in_basis_points / _BASIS_POINTS_TO_DECIMAL_CONVERSION_FACTOR
    )
    daily_portfolio_return_and_transaction_count_df[
        PORTFOLIO_DAILY_RETURN_AFTER_T_COST_COLUMN_NAME
    ] = (
        daily_portfolio_return_df[PORTFOLIO_DAILY_RETURN_COLUMN_NAME]
        - daily_portfolio_return_and_transaction_count_df[
            _PORTFOLIO_DAILY_T_COST_COLUMN_NAME
        ]
    )
    return daily_portfolio_return_and_transaction_count_df[
        [PORTFOLIO_DAILY_RETURN_AFTER_T_COST_COLUMN_NAME]
    ].fillna(0)


def calculate_annualized_portfolio_return(
    daily_portfolio_return_df: pd.DataFrame,
) -> float:
    daily_portfolio_return_df[
        _PORTFOLIO_GROSS_DAILY_RETURN_AFTER_T_COST_COLUMN_NAME
    ] = (daily_portfolio_return_df[PORTFOLIO_DAILY_RETURN_AFTER_T_COST_COLUMN_NAME] + 1)
    total_gross_return = daily_portfolio_return_df[
        _PORTFOLIO_GROSS_DAILY_RETURN_AFTER_T_COST_COLUMN_NAME
    ].prod()
    num_days_in_performance_period = len(daily_portfolio_return_df)
    annualized_gross_return = total_gross_return ** (
        _ANNUALIZATION_FACTOR_IN_TRADING_DAYS / num_days_in_performance_period
    )
    annualized_return = annualized_gross_return - 1
    return annualized_return


def get_portfolio_performance_period_start(
    daily_portfolio_returns_df: pd.DataFrame,
) -> datetime:
    return daily_portfolio_returns_df.index[0]


def get_portfolio_performance_period_end(
    daily_portfolio_returns_df: pd.DataFrame,
) -> datetime:
    return daily_portfolio_returns_df.index[-1]


def calculate_annualized_portfolio_volatility(
    daily_portfolio_return_df: pd.DataFrame,
) -> float:
    daily_volatility = daily_portfolio_return_df[
        PORTFOLIO_DAILY_RETURN_AFTER_T_COST_COLUMN_NAME
    ].std()
    return daily_volatility * np.sqrt(_ANNUALIZATION_FACTOR_IN_TRADING_DAYS)


def calculate_annualized_sharpe_ratio(
    annualized_return: float,
    annualized_volatility: float,
) -> float:
    excess_return = annualized_return - _RISK_FREE_RATE_IN_DECIMAL
    return excess_return / annualized_volatility


# TODO convert returns to decimal higher up stream
# to prevent repeated conversions
def get_portfolio_performance_result(
    trade_returns_for_all_tickers_df: pd.DataFrame,
) -> Mapping[str, float]:
    trade_returns_for_all_tickers_df = trade_returns_for_all_tickers_df / 100
    daily_portfolio_return_df = (
        calculate_daily_return_on_employed_capital_after_t_costs(
            trade_returns_for_all_tickers_df, _ONE_WAY_T_COST_IN_BASIS_POINTS
        )
    )
    annualized_return = calculate_annualized_portfolio_return(
        daily_portfolio_return_df.copy()
    )
    annualized_volatility = calculate_annualized_portfolio_volatility(
        daily_portfolio_return_df.copy()
    )
    portfolio_performance_result = PortfolioPerformanceResult(
        annualized_return=annualized_return,
        annualized_volatility=annualized_volatility,
        annualized_sharpe_ratio=calculate_annualized_sharpe_ratio(
            annualized_return, annualized_volatility
        ),
        performance_period_start=get_portfolio_performance_period_start(
            daily_portfolio_return_df
        ),
        performance_period_end=get_portfolio_performance_period_end(
            daily_portfolio_return_df
        ),
    )
    return portfolio_performance_result


if __name__ == "__main__":
    # NOTE dates are the one year period following the two year period
    # that was screened for cointegration
    z_score_window_in_calendar_days = 15
    num_pairs = 3
    start_date = datetime(2024, 1, 1)
    # TODO the +10 is a hack to account for calendar vs trading days. Fix this
    start_date_adj_for_z_score_window = start_date - timedelta(
        days=z_score_window_in_calendar_days + 10
    )
    end_date = datetime(2024, 12, 31)
    path_to_ticker_list = Path("data/russell_3000_constituents.csv")
    pairs_and_hurst_exponents = get_tmp_hurst_exps_from_disk()
    pairs_and_hurst_exponents_top_n = pairs_and_hurst_exponents.sort_values(
        by="hurst_exp"
    ).head(num_pairs)
    all_tickers_price_history_dict = read_stock_price_history_into_dict(
        path_to_ticker_list
    )
    trade_returns_for_all_tickers_df = pd.DataFrame()
    for ticker_one, ticker_two in pairs_and_hurst_exponents_top_n["ticker_pair"]:
        pair_price_history_df = get_pair_price_history_df_algined_on_date(
            ticker_one,
            ticker_two,
            start_date_adj_for_z_score_window,
            end_date,
            all_tickers_price_history_dict,
        )
        gamma = calculate_historical_gamma(pair_price_history_df, start_date, end_date)
        spread_series = calculate_spread(
            pair_price_history_df, gamma, start_date_adj_for_z_score_window, end_date
        )
        z_scored_spread_series = calculate_rolling_zscore(
            spread_series,
            start_date,
            end_date,
            z_score_window_in_days=z_score_window_in_calendar_days,
        )
        breakpoint()
        trade_signals_df = create_trade_signals_from_z_scored_spread(
            ticker_one,
            ticker_two,
            z_scored_spread_series,
            _EXIT_THRESHOLD_ABSOLUTE_TOLERANCE,
        )
        breakpoint()
        price_returns_df = create_returns_from_price_history(pair_price_history_df)
        trade_returns_for_tickers_in_pair_df = (
            calculate_trade_returns_for_tickers_in_pair(
                ticker_one, ticker_two, trade_signals_df, price_returns_df
            )
        )
        trade_returns_for_all_tickers_df = pd.concat(
            [trade_returns_for_all_tickers_df, trade_returns_for_tickers_in_pair_df],
            axis=1,
        )
    portfolio_performance_result = get_portfolio_performance_result(
        trade_returns_for_all_tickers_df
    )
    print(portfolio_performance_result)
