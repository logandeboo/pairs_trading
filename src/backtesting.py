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
import sys

# TODO this is a temporary workaround
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# TODO add file to hold common funcs
from src.pair_selection import (
    read_stock_price_history_into_dict,
    calculate_spread,
    calculate_historical_gamma,
    get_pair_price_history_df_algined_on_date,
    create_returns_from_price_history,
)


TRADE_SIDE_COLUMN_NAME_SUFFIX = "_trade_side"


# TODO consider triggering condition as the spread converges *back*
# within range instead of upon exit
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


def create_trade_signals_from_spread_rolling_z_score(
    ticker_pair: tuple[str, str],
    spread_z_score_series: pd.Series,
) -> pd.DataFrame:
    ticker_one = ticker_pair[0]
    ticker_two = ticker_pair[1]
    ticker_one_weight_column_name = ticker_one + TRADE_SIDE_COLUMN_NAME_SUFFIX
    ticker_two_weight_column_name = ticker_two + TRADE_SIDE_COLUMN_NAME_SUFFIX

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


def calculate_z_score_from_coint_period(
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
    ticker_one_trade_side_column_name = ticker_one + TRADE_SIDE_COLUMN_NAME_SUFFIX
    ticker_two_trade_side_column_name = ticker_two + TRADE_SIDE_COLUMN_NAME_SUFFIX
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


# This method of calculating portfolio returns assumes that positions are
# always equal to num_pairs / portfolio_value. In practice, this would entail adjusting
# the weight of open positions every time a position is closed and a return is realized.
# Otherwise, the portfolio would not be fully invested and thus returns
# would not compound the way they are calculated below
def calculate_portfolio_compounded_return(
    trade_returns_for_all_tickers_df: pd.DataFrame,
) -> pd.DataFrame:
    portfolio_returns_df = pd.DataFrame()
    trade_returns_for_all_tickers_df = trade_returns_for_all_tickers_df / len(
        trade_returns_for_all_tickers_df.columns
    )
    portfolio_returns_df["portfolio_daily_return_pct"] = (
        trade_returns_for_all_tickers_df.sum(axis=1)
    )
    portfolio_returns_df["portfolio_daily_return_decimal"] = (
        portfolio_returns_df["portfolio_daily_return_pct"] / 100
    )
    portfolio_returns_df["gross_daily_return"] = (
        1 + portfolio_returns_df["portfolio_daily_return_decimal"]
    )
    portfolio_returns_df["compounded_daily_return"] = portfolio_returns_df[
        "gross_daily_return"
    ].cumprod()
    return portfolio_returns_df[
        ["portfolio_daily_return_pct", "compounded_daily_return"]
    ]

def calculate_number_of_trades_per_day(curr: pd.DataFrame) -> pd.Series:
    prev = curr.shift(1).fillna(0)

    buys = (curr != 0) & (prev == 0)
    sells = (prev != 0) & (curr == 0)
    # Sells are shifted up on day so t cost can be subtracted 
    # from the last day the position was active
    transaction_count_df = buys.sum(axis=1) + sells.shift(-1).sum(axis=1)
    num_trades_closed_on_last_day_of_period = (curr.iloc[-1] != 0).sum()
    transaction_count_df.iloc[-1] += num_trades_closed_on_last_day_of_period
    return transaction_count_df


# Calculates total portfolio return on employed capital for equal dollar weight trades.
# Σ(active trade returns) / number of active trades
# For each day, the portfolio return is calculated as the sum of returns on open trades divided
# by the number of open positions. E.g., on day t, if there are two open trades with returns of 
# .05% and 1.2%, the portfolio return is (0.0005 + 0.012) / 2
def calculate_return_on_employed_captial(
    trade_returns_for_all_tickers_df: pd.DataFrame,
) -> float:
    trade_returns_for_all_tickers_df['portfolio_daily_return'] = trade_returns_for_all_tickers_df.sum(axis=1) / trade_returns_for_all_tickers_df.astype(bool).sum(axis=1)
    trade_returns_for_all_tickers_df['portfolio_gross_return'] = trade_returns_for_all_tickers_df['portfolio_daily_return'] + 1
    return trade_returns_for_all_tickers_df['portfolio_gross_return'].cumprod().iloc[-1] - 1
    
# def calculate_portfolio_performance(
#     trade_returns_for_all_tickers_df: pd.DataFrame,
# ) -> Mapping[str, float]:
#     performance = {}
#     trade_returns_for_all_tickers_df = trade_returns_for_all_tickers_df / 100
#     performancecalculate_return_on_employed_captial(trade_returns_for_all_tickers_df)

    
    
    



if __name__ == "__main__":
    # NOTE dates are the one year period following the two year period
    # that was screened for cointegration
    z_score_window_in_calendar_days = 15
    num_pairs = 2
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
    for pair in pairs_and_hurst_exponents_top_n["ticker_pair"]:
        ticker_one = pair[0]
        ticker_two = pair[1]
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
        trade_signals_df = create_trade_signals_from_spread_rolling_z_score(
            pair, z_scored_spread_series
        )
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
    return_on_employed_capital = calculate_return_on_employed_captial(
        trade_returns_for_all_tickers_df
    )
    print(return_on_employed_capital)