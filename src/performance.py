from datetime import datetime
from typing import NamedTuple
import pandas as pd
import numpy as np
from typing import Mapping, Collection
from src.simulation.simulation import get_simulated_returns_by_portfolio_tickers_df

PORTFOLIO_DAILY_RETURN_COLUMN_NAME = "portfolio_daily_return"
_ANNUALIZATION_FACTOR_IN_TRADING_DAYS = 252
_RISK_FREE_RATE_IN_DECIMAL = 0.04
_BASIS_POINTS_TO_DECIMAL_CONVERSION_FACTOR = 10_000
_ONE_WAY_T_COST_IN_BASIS_POINTS = 10
_PORTFOLIO_GROSS_DAILY_RETURN_AFTER_T_COST_COLUMN_NAME = (
    "portfolio_gross_daily_return_after_t_cost"
)
_PORTFOLIO_DAILY_T_COST_COLUMN_NAME = "portfolio_daily_t_cost"
PORTFOLIO_DAILY_RETURN_AFTER_T_COST_COLUMN_NAME = "portfolio_daily_return_after_t_cost"
DAILY_TRANSACTION_COUNT_COLUMN_NAME = "transaction_count"


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
def calculate_net_daily_portfolio_return(
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


def get_portfolio_performance_result(
    start_date_for_simulation: datetime,
    end_date_for_simulation: datetime,
    pairs_for_portfolio_simulation: Collection[tuple[str, str, float]],
    all_tickers_price_history_df: pd.DataFrame,
    *,
    spread_z_score_rolling_window_in_trading_days: int,
) -> Mapping[str, float]:
    returns_by_ticker_df = get_simulated_returns_by_portfolio_tickers_df(
        start_date_for_simulation,
        end_date_for_simulation,
        pairs_for_portfolio_simulation,
        all_tickers_price_history_df,
        spread_z_score_rolling_window_in_trading_days,
    )
    daily_portfolio_return_df = calculate_net_daily_portfolio_return(
        returns_by_ticker_df, _ONE_WAY_T_COST_IN_BASIS_POINTS
    )
    annualized_return = calculate_annualized_portfolio_return(
        daily_portfolio_return_df.copy()
    )
    annualized_volatility = calculate_annualized_portfolio_volatility(
        daily_portfolio_return_df.copy()
    )
    return PortfolioPerformanceResult(
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
