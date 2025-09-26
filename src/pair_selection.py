import pandas as pd
import yfinance as yf
from collections.abc import Collection
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import numpy as np
from datetime import datetime, date, timedelta
import os
import matplotlib.pyplot as plt
import statsmodels.api as sm
import requests
from pathlib import Path


# for ticker and date range
# if data on prem --> return
# if data is only partially on prem
#   query the missing portion and write it to db
#   then read it from on prem


def read_adj_close_history_df_from_on_prem(
    path: str, start_date: datetime, end_date: datetime
) -> pd.DataFrame:
    price_history_df = pd.read_csv(path, index_col=0, parse_dates=True)
    return price_history_df[
        (price_history_df.index >= start_date) & (price_history_df.index <= end_date)
    ]


def get_stocks_adj_close_history_df(
    tickers: list[str], start_date: datetime, end_date: datetime
) -> tuple[pd.Series, pd.Series]:
    all_tickers_price_history_df = pd.DataFrame()
    for ticker in tickers:
        path_to_ticker_data = Path(f"data/adj_close_price_data/{ticker}.csv")
        ticker_price_history_df = read_adj_close_history_df_from_on_prem(
            path_to_ticker_data, start_date, end_date
        )
        all_tickers_price_history_df = pd.concat(
            [all_tickers_price_history_df, ticker_price_history_df], axis=1
        )
    return all_tickers_price_history_df


def get_benchmark_adj_close_history_df(
    benchmark_ticker: str, start_date: datetime, end_date: datetime
) -> pd.DataFrame:
    path_to_benchmark_data = Path(
        f"data/adj_close_price_data_benchmarks/{benchmark_ticker}.csv"
    )
    return read_adj_close_history_df_from_on_prem(
        path_to_benchmark_data, start_date, end_date
    )


def calculate_regression_coefficient(Y: pd.Series, X: pd.Series, x_label: str) -> float:
    X = sm.add_constant(X)
    results = sm.OLS(Y, X).fit()
    return float(results.params[x_label])


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
# current parameter values are from chatgpt
# also corroborated here https://blog.quantinsti.com/johansen-test-cointegration-building-stationary-portfolio/
def is_pair_johansen_cointegrated(
    stock_and_benchmark_price_history_df: pd.DataFrame, ticker_one: str, ticker_two: str
) -> bool:
    pair_price_history_df = stock_and_benchmark_price_history_df[
        [ticker_one, ticker_two]
    ]
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


def create_returns_from_price_history(price_history_df: pd.DataFrame) -> float:
    pair_price_returns_df = price_history_df.pct_change() * 100
    return pair_price_returns_df


def are_pair_betas_close_enough(
    ticker_one: str,
    ticker_two: str,
    benchmark_ticker: str,
    stock_and_benchmark_price_history_df: pd.DataFrame,
) -> bool:
    beta_absolute_difference_threshold = 0.3
    ticker_one_price_series = stock_and_benchmark_price_history_df[ticker_one]
    ticker_two_price_series = stock_and_benchmark_price_history_df[ticker_two]
    benchmark_price_series = stock_and_benchmark_price_history_df[benchmark_ticker]
    ticker_one_beta = calculate_regression_coefficient(
        ticker_one_price_series, benchmark_price_series, benchmark_ticker
    )
    ticker_two_beta = calculate_regression_coefficient(
        ticker_two_price_series, benchmark_price_series, benchmark_ticker
    )
    is_close_enough = np.isclose(
        ticker_one_beta,
        ticker_two_beta,
        atol=beta_absolute_difference_threshold,
        rtol=0,
    )
    return bool(is_close_enough)


def get_stock_and_benchmark_prices_df_algined_on_date(
    ticker_one: str,
    ticker_two: str,
    benchmark_ticker: str,
    beta_calculation_period_start: datetime,
    end_date: datetime,
) -> pd.DataFrame:
    pair_price_history_df = get_stocks_adj_close_history_df(
        [ticker_one, ticker_two], beta_calculation_period_start, end_date
    )
    benchmark_price_history_df = get_benchmark_adj_close_history_df(
        benchmark_ticker, beta_calculation_period_start, end_date
    )
    return benchmark_price_history_df.merge(
        pair_price_history_df, how="inner", left_index=True, right_index=True
    ).dropna()


# NOTE: This methodology (ex beta filter) came from Caldeira & Caldeira 2013. Paper is in references folder
def is_pair_tradable(
    ticker_one: str,
    ticker_two: str,
    end_date: datetime,
    benchmark_ticker: str,
    beta_estimation_window_in_days: int,
) -> bool:
    beta_calculation_period_start = end_date - timedelta(
        days=beta_estimation_window_in_days
    )
    try:
        stock_and_benchmark_price_history_df = (
            get_stock_and_benchmark_prices_df_algined_on_date(
                ticker_one,
                ticker_two,
                benchmark_ticker,
                beta_calculation_period_start,
                end_date,
            )
        )
        if not are_pair_betas_close_enough(
            ticker_one,
            ticker_two,
            benchmark_ticker,
            stock_and_benchmark_price_history_df,
        ):
            return False
        ticker_one_price_series = stock_and_benchmark_price_history_df[ticker_one]
        ticker_two_price_series = stock_and_benchmark_price_history_df[ticker_two]
        if not is_price_series_integrated_of_order_one(ticker_one_price_series):
            return False
        if not is_price_series_integrated_of_order_one(ticker_two_price_series):
            return False
        if not is_pair_engle_granger_cointegrated(
            ticker_one_price_series, ticker_two_price_series
        ):
            return False
        if not is_pair_johansen_cointegrated(
            stock_and_benchmark_price_history_df, ticker_one, ticker_two
        ):
            return False
        return True
    except Exception as e:
        print(f"Unable to evaluate pair {ticker_one + '/' + ticker_two}")
        print(e)
        return False


def calculate_historical_gamma(
    pair_price_history_df: pd.DataFrame,
    in_sample_start_date: datetime,
    in_sample_end_date: datetime,
) -> float:
    """
    Estimate gamma via OLS and return spread = P1 - gamma * P2
    """
    date_mask = (pair_price_history_df.index >= in_sample_start_date) & (
        pair_price_history_df.index <= in_sample_end_date
    )
    pair_price_history_in_sample_df = pair_price_history_df[date_mask]
    ticker1_price_series = pair_price_history_in_sample_df.iloc[:, 0]
    ticker2_price_series = pair_price_history_in_sample_df.iloc[:, 1]

    # Add constant to allow intercept in regression
    X = sm.add_constant(ticker2_price_series)
    model = sm.OLS(ticker1_price_series, X).fit()

    return float(model.params[1])


def calculate_spread(
    pair_price_history_df: pd.DataFrame,
    gamma: float,
    start_date: datetime,
    end_date: datetime,
) -> pd.Series:
    date_mask = (pair_price_history_df.index >= start_date) & (
        pair_price_history_df.index <= end_date
    )
    pair_price_history_out_of_sample_df = pair_price_history_df[date_mask]
    ticker1_price_series = pair_price_history_out_of_sample_df.iloc[:, 0]
    ticker2_price_series = pair_price_history_out_of_sample_df.iloc[:, 1]

    return ticker1_price_series - gamma * ticker2_price_series


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


# TODO add time series of prices on this graph for additional clarity
def plot_zscore_zeries(zscore_series: pd.Series):
    plt.figure(figsize=(14, 6))
    plt.plot(zscore_series.index, zscore_series.values, label="Z-Score", color="blue")

    # Horizontal lines for 1 and 2 standard deviations
    for level in [1, 2]:
        plt.axhline(
            level,
            color="gray",
            linestyle="--",
            linewidth=1,
            label=f"+{level}σ" if level == 1 else None,
        )
        plt.axhline(
            -level,
            color="gray",
            linestyle="--",
            linewidth=1,
            label=f"-{level}σ" if level == 1 else None,
        )

    # Only show one label per line for legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.title("Trailing Z-Score of Spread (Out-of-Sample)")
    plt.xlabel("Date")
    plt.ylabel("Z-Score")
    plt.grid(True)
    plt.show()


def get_ticker_list(path_to_ticker_list: Path) -> list[tuple[str, str]]:
    ticker_df = pd.read_csv(path_to_ticker_list)
    return ticker_df["Ticker"].to_list()


def get_ticker_pairs() -> list[tuple]:
    path_to_ticker_list = Path("data/russell_3000_constituents.csv")
    tickers = get_ticker_list(path_to_ticker_list)
    pairs = []
    for ticker_one in tickers:
        for ticker_two in tickers:
            if ticker_one == ticker_two:
                continue
            pairs.append((ticker_one, ticker_two))
    return pairs


# NOTE For practical reasons all data this function requires must be written to
# target folders beforehand

# TODO should eventually create a generic abstraction that takes
# a universe (list of tickers) and returns a collection of tradable pairs.
# But that computation is time consuming and I already have a list of candidate pairs
# so I'm going to start there for now.


# TODO the benchmark constituents are not PIT so there is technically survivorship bias

# TODO add a check that ensures the series being used to calculate beta
# are of a certain length
if __name__ == "__main__":
    valid_pairs = []
    seen_pairs = []
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2023, 12, 31)
    benchmark_ticker = "IWV"
    beta_estimation_window_in_calendar_days = 365 * 3 + 1
    valid_pairs_output_path = Path(f"data/valid_pairs_{datetime.now()}.csv")

    pairs = get_ticker_pairs()

    for pair in pairs:
        if sorted(pair) in seen_pairs:
            continue
        ticker_one = pair[0]
        ticker_two = pair[1]
        if is_pair_tradable(
            ticker_one,
            ticker_two,
            start_date,
            end_date,
            benchmark_ticker,
            beta_estimation_window_in_calendar_days,
        ):
            valid_pairs.append(pair)
            pd.DataFrame({"Valid Pairs": [pair]}).to_csv(
                valid_pairs_output_path,
                mode="a",
                header=False,
                index=False,
            )
        seen_pairs.append(sorted(pair))
