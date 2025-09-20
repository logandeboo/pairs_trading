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


# for ticker and date range
# if data on prem --> return
# if data is only partially on prem
#   query the missing portion and write it to db
#   then read it from on prem


def get_path_to_adj_close_history_cache(ticker: str) -> str:
    return f"../data/adj_close_price_data/{ticker}.csv"


def get_price_history_from_yf(
    ticker: str,
    start_date: datetime,
    end_date: datetime,
) -> pd.DataFrame:
    price_column_name_suffix = "_adj_close"
    price_history_df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        progress=False,
        auto_adjust=True,
    )["Close"]
    price_history_df.columns = [
        f"{col_name + price_column_name_suffix}"
        for col_name in price_history_df.columns
    ]
    return price_history_df


def get_updated_existing_price_history_df(
    ticker: str,
    existing_price_history_df: pd.DataFrame,
    start_date: datetime,
    end_date: datetime,
) -> pd.DataFrame:
    path_to_ticker_data = get_path_to_adj_close_history_cache(ticker)
    oldest_existing_day = existing_price_history_df.index[0]
    newest_existing_day = existing_price_history_df.index[-1]

    if start_date < oldest_existing_day:
        missing_older_prices_df = get_price_history_from_yf(
            ticker, start_date, oldest_existing_day - timedelta(days=1)
        )
        existing_price_history_df = pd.concat(
            [missing_older_prices_df, existing_price_history_df]
        )

    if end_date > newest_existing_day:
        missing_newer_prices_df = get_price_history_from_yf(
            ticker, newest_existing_day + timedelta(days=1), end_date
        )
        existing_price_history_df = pd.concat(
            [existing_price_history_df, missing_newer_prices_df]
        )
    existing_price_history_df = existing_price_history_df[
        ~existing_price_history_df.index.duplicated(keep="last")
    ]
    existing_price_history_df.to_csv(path_to_ticker_data, index=False)
    return existing_price_history_df


def get_cached_adj_close_price_history(
    ticker: str,
    start_date: datetime,
    end_date: datetime,
) -> pd.DataFrame:
    path_to_ticker_data = get_path_to_adj_close_history_cache(ticker)
    if os.path.exists(path_to_ticker_data):
        existing_price_history_df = pd.read_csv(path_to_ticker_data)
        return get_updated_existing_price_history_df(
            ticker,
            existing_price_history_df,
            start_date,
            end_date,
        )
    else:
        missing_prices_df = get_price_history_from_yf(ticker, start_date, end_date)
        missing_prices_df.to_csv(path_to_ticker_data, index=False)


def get_adj_close_history_df_NEW(
    tickers: tuple[str, str], start_date: datetime, end_date: datetime
) -> pd.DataFrame:
    historical_adj_close_all_tickers_df = pd.DataFrame()
    for ticker in tickers:
        historical_adj_close_single_ticker_df = get_cached_adj_close_price_history(
            ticker
        )
        historical_adj_close_all_tickers_df = pd.concat(
            [
                historical_adj_close_all_tickers_df,
                historical_adj_close_single_ticker_df,
            ],
            axis=1,
        )
    return historical_adj_close_all_tickers_df


def get_adj_close_history_df(
    tickers: list[str], start_date: datetime, end_date: datetime
) -> pd.DataFrame:
    price_column_name_suffix = "_price"
    prices_df = yf.download(
        tickers, start=start_date, end=end_date, progress=False, auto_adjust=True
    )["Close"]
    prices_df.columns = [
        f"{col_name + price_column_name_suffix}" for col_name in prices_df.columns
    ]
    return prices_df.dropna()


def create_returns_from_price_history(price_history_df: pd.DataFrame) -> float:
    pair_price_returns_df = price_history_df.pct_change() * 100
    pair_price_returns_df.columns = pair_price_returns_df.columns.str.replace(
        r"_price$", "_1d_returns", regex=True
    )
    return pair_price_returns_df


def calculate_regression_coefficient(Y: pd.Series, X: pd.Series, x_label: str) -> float:
    X = sm.add_constant(X)
    results = sm.OLS(Y, X).fit()
    return float(results.params[x_label])


def calculate_historical_beta_at_date(ticker: str, cur_date: datetime) -> float:
    benchmark_ticker = "IWV"  # russell 3000 etf
    beta_calculation_window_in_days = (
        365 * 3 + 1
    )  # adding one to get returns for full period
    beta_calculation_period_start = cur_date - timedelta(
        days=beta_calculation_window_in_days
    )
    prices_df = get_adj_close_history_df(
        [ticker, benchmark_ticker],
        beta_calculation_period_start,
        cur_date,
    )
    returns_df = create_returns_from_price_history(prices_df).dropna()
    returns_df.columns = returns_df.columns.str.replace(r"_1d_returns", "", regex=True)
    return calculate_regression_coefficient(
        returns_df[ticker], returns_df[benchmark_ticker], benchmark_ticker
    )


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
def is_pair_johansen_cointegrated(pair_price_history_df: pd.DataFrame) -> bool:
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


def are_historical_betas_close_enough(
    as_of_date: datetime, ticker_one: str, ticker_two: str
) -> bool:
    beta_absolute_difference_threshold = 0.3
    ticker_one_beta = calculate_historical_beta_at_date(ticker_one, as_of_date)
    ticker_two_beta = calculate_historical_beta_at_date(ticker_two, as_of_date)
    is_close_enough = np.isclose(
        ticker_one_beta,
        ticker_two_beta,
        atol=beta_absolute_difference_threshold,
        rtol=0,
    )
    return bool(is_close_enough)


# NOTE: This methodology (ex beta filter) came from Caldeira & Caldeira 2013. Paper is in references folder
def is_pair_tradable(
    ticker_one: str,
    ticker_two: str,
    start_date: datetime,
    end_date: datetime,
    beta_estimation_window_in_days: int,
) -> bool:
    beta_calculation_period_start = end_date - timedelta(
        days=beta_estimation_window_in_days
    )
    try:
        pair_price_history_df = get_adj_close_history_df(
            ticker_one, ticker_two, beta_calculation_period_start, end_date
        )
        if not are_historical_betas_close_enough(end_date, ticker_one, ticker_two):
            print(f"Pair {ticker_one + '/' + ticker_two} betas are not close enough")
            return False
        ticker1_price_series = pair_price_history_df.iloc[:, 0]
        ticker2_price_series = pair_price_history_df.iloc[:, 1]
        if not is_price_series_integrated_of_order_one(ticker1_price_series):
            print(
                f"Pair {ticker_one + '/' + ticker_two} first ticker series is not I(1)"
            )
            return False
        if not is_price_series_integrated_of_order_one(ticker2_price_series):
            print(
                f"Pair {ticker_one + '/' + ticker_two} second ticker series is not I(1)"
            )
            return False
        if not is_pair_engle_granger_cointegrated(
            ticker1_price_series, ticker2_price_series
        ):
            print(
                f"Pair {ticker_one + '/' + ticker_two} is not engle granger cointegrated"
            )
            return False
        if not is_pair_johansen_cointegrated(pair_price_history_df):
            print(f"Pair {ticker_one + '/' + ticker_two} is not johansen cointegrated")
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
    Estimate gamma via OLS and return the spread = P1 - gamma * P2
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


def get_ticker_pairs() -> list[tuple]:
    ticker_df = pd.read_csv("../data/russell_3000_constituents.csv")
    tickers = ticker_df["Ticker"].to_list()
    pairs = []
    for ticker_one in tickers:
        for ticker_two in tickers:
            if ticker_one == ticker_two:
                continue
            pairs.append((ticker_one, ticker_two))
    return pairs


# TODO should eventually create a generic abstraction that takes
# a universe (list of tickers) and returns a collection of tradable pairs.
# But that computation is time consuming and I already have a list of candidate pairs
# so I'm going to start there for now.


if __name__ == "__main__":
    valid_pairs = []
    seen_pairs = []
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2023, 12, 31)

    pairs = get_ticker_pairs()

    for pair in pairs:
        if sorted(pair) in seen_pairs:
            continue
        if is_pair_tradable(pair, start_date, end_date):
            print("VALID:", pair)
            valid_pairs.append(pair)
            pd.DataFrame({"Valid Pairs": [pair]}).to_csv(
                "../data/valid_sp500_pairs.csv",
                mode="a",  # append mode
                header=False,  # don’t write header again
                index=False,
            )
        seen_pairs.append(sorted(pair))
