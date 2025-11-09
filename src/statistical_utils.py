from __future__ import annotations
import pandas as pd
from datetime import datetime
import numpy as np
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from src.time_series_utils import (
    add_us_trading_days_to_date,
    filter_series_or_df_by_dates,
)
from src.stock import Stock
from src.risk.risk_factor import RiskFactor
from src.risk.risk_factors import RISK_FREE_RATE, RISK_FREE_RATE_COLUMN_NAME

_TICKER_ONE_INDEX = 0
_TICKER_TWO_INDEX = 1


def calculate_gamma(
    pair_price_history_df: pd.DataFrame,
) -> float:
    ticker1_price_series = pair_price_history_df.iloc[:, _TICKER_ONE_INDEX]
    ticker2_price_series = pair_price_history_df.iloc[:, _TICKER_TWO_INDEX]
    return calculate_regression_coefficient(ticker2_price_series, ticker1_price_series)


def get_pair_spread_series(
    pair_price_history_df: pd.DataFrame,
) -> pd.Series:
    gamma = calculate_gamma(pair_price_history_df)
    ticker1_price_series = pair_price_history_df.iloc[:, _TICKER_ONE_INDEX]
    ticker2_price_series = pair_price_history_df.iloc[:, _TICKER_TWO_INDEX]
    return ticker1_price_series - gamma * ticker2_price_series


def calculate_regression_coefficient(
    x_series: pd.Series[float], y_series: pd.Series[float]
) -> float:
    slope, _ = np.polyfit(x_series.values, y_series.values, deg=1)
    return float(slope)


def create_daily_returns(price_history_df: pd.DataFrame) -> float:
    pair_price_returns_df = price_history_df.pct_change() * 100
    return pair_price_returns_df


# Bui & Slepaczuk methodology
# TODO underwrite num_lags
# TODO contradiction between signature and return np.nan in second line
def calculate_generalized_hurst_exponent_q1(series: pd.Series) -> float:
    if len(series) < 100:
        return np.nan
    data = np.array(series.dropna())
    max_tau = len(data) // 4
    min_lag = 2
    num_lags = 20
    tau_values = np.unique(
        np.logspace(np.log10(min_lag), np.log10(max_tau), num_lags).astype(int)
    )
    # Calculate K_q(τ) for each lag value
    K_q_values = []
    for tau in tau_values:
        increments = np.abs(data[tau:] - data[:-tau])
        K_q_tau = np.mean(increments)
        K_q_values.append(K_q_tau)
    K_q_values = np.array(K_q_values)
    # Fit power law relationship
    log_tau = np.log(tau_values)
    log_K_q = np.log(K_q_values)
    valid_mask = np.isfinite(log_K_q) & np.isfinite(log_tau)
    if np.sum(valid_mask) < 3:
        return np.nan
    log_tau_valid = log_tau[valid_mask]
    log_K_q_valid = log_K_q[valid_mask]
    # Linear regression: log(K_q) = H * log(τ) + intercept
    hurst_exponent, _ = np.polyfit(log_tau_valid, log_K_q_valid, 1)
    return hurst_exponent


def is_price_series_integrated_of_order_one(
    price_series: pd.Series, pvalue_threshold: float = 0.05
) -> bool:
    p_level = adfuller(price_series)[1]
    # Test if series is stationary
    if p_level < pvalue_threshold:
        return False

    # Test if series is stationary after applying one difference
    p_diff = adfuller(price_series.diff().dropna())[1]
    return True if p_diff < pvalue_threshold else False


def is_pair_engle_granger_cointegrated(
    ticker1_price_series: pd.Series,
    ticker2_price_series: pd.Series,
    pvalue_threshold: float = 0.05,
) -> bool:
    _, pvalue, _ = coint(ticker1_price_series, ticker2_price_series)
    return True if pvalue < pvalue_threshold else False


# TODO underwrite det_order and k_ar_diff parameter values
# current parameter values are from
# https://blog.quantinsti.com/johansen-test-cointegration-building-stationary-portfolio/
def is_pair_johansen_cointegrated(
    ticker_one_price_series: pd.Series, ticker_two_price_series: pd.Series
) -> bool:
    pair_price_history_df = pd.concat(
        [ticker_one_price_series, ticker_two_price_series], axis=1, join="inner"
    ).dropna()
    johansen_cointegration_result = coint_johansen(
        pair_price_history_df.values, det_order=0, k_ar_diff=1
    )
    index_of_95_pct_confidence_level_critical_value = 1
    index_of_trace_stat = 0
    trace_stat = johansen_cointegration_result.lr1[index_of_trace_stat]
    critical_value = johansen_cointegration_result.cvt[
        0, index_of_95_pct_confidence_level_critical_value
    ]
    return trace_stat > critical_value


def calculate_trailing_zscore(
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


def get_pair_spread_rolling_z_score_series(
    backtest_start_date_adj_for_z_score_rolling_window: datetime,
    spread_rolling_z_score_series_end_date: datetime,
    pair_price_history_df: pd.DataFrame,
    *,
    z_score_window_in_trading_days: int,
) -> pd.Series:
    spread_series = get_pair_spread_series(
        pair_price_history_df,
        backtest_start_date_adj_for_z_score_rolling_window,
        spread_rolling_z_score_series_end_date,
    )
    spread_rolling_mean_series = spread_series.rolling(
        window=z_score_window_in_trading_days
    ).mean()
    spread_rolling_std_series = spread_series.rolling(
        window=z_score_window_in_trading_days
    ).std()
    spread_rolling_z_score_series = (
        spread_series - spread_rolling_mean_series
    ) / spread_rolling_std_series
    spread_rolling_z_score_series_start_date = add_us_trading_days_to_date(
        backtest_start_date_adj_for_z_score_rolling_window,
        offset_in_us_trading_days=z_score_window_in_trading_days,
    )
    return filter_series_or_df_by_dates(
        spread_rolling_z_score_series_start_date,
        spread_rolling_z_score_series_end_date,
        spread_rolling_z_score_series,
    )


def get_excess_stock_return_df(
    start_date: datetime,
    end_date: datetime,
    stock: Stock,
) -> pd.Series:
    stock_daily_returns_df = filter_series_or_df_by_dates(
        start_date, end_date, stock.daily_returns_df
    )
    risk_free_daily_returns_df = filter_series_or_df_by_dates(
        start_date, end_date, RISK_FREE_RATE.returns_df
    )
    joined_df = stock_daily_returns_df.join(risk_free_daily_returns_df, how="inner")
    stock_excess_return_column_name = f"{stock.ticker}_excess_return"
    joined_df[stock_excess_return_column_name] = (
        joined_df[stock.ticker] - joined_df[RISK_FREE_RATE_COLUMN_NAME]
    )
    return joined_df[[stock_excess_return_column_name]]


def get_stock_exposure_to_risk_factor(
    start_date: datetime, end_date: datetime, stock: Stock, risk_factor: RiskFactor
) -> float:
    excess_stock_returns_df = get_excess_stock_return_df(start_date, end_date, stock)
    joined_df = risk_factor.returns_df.join(excess_stock_returns_df, how="inner")
    return calculate_regression_coefficient(
        joined_df.iloc[:, 0],
        joined_df.iloc[:, 1],
    )
