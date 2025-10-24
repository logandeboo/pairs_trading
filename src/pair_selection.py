import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import numpy as np
from datetime import datetime
from pathlib import Path
from itertools import combinations
from typing import Sequence, Mapping, Union, Collection
import pickle
from pandas.tseries.holiday import USFederalHolidayCalendar, GoodFriday
from pandas.tseries.offsets import CustomBusinessDay
from src.time_series_utils import (
    filter_price_history_series_or_df_by_date,
    get_pair_price_history_df_algined_on_date
)


_COLUMBUS_DAY_HOLIDAY_STRING = "Columbus Day"
_VETRANS_DAY_HOLIDAY_STRING = "Veterans Day"
_ONE_YEAR_IN_TRADING_DAYS = 252


class USTradingCalendar(USFederalHolidayCalendar):
    rules = [
        r
        for r in USFederalHolidayCalendar.rules
        if r.name not in [_COLUMBUS_DAY_HOLIDAY_STRING, _VETRANS_DAY_HOLIDAY_STRING]
    ] + [GoodFriday]


def subtract_n_us_trading_days_from_date(
    date: datetime, offset_in_us_trading_days: int
) -> datetime:
    trading_day = CustomBusinessDay(calendar=USTradingCalendar())
    return (
        pd.Timestamp(date) - offset_in_us_trading_days * trading_day
    ).to_pydatetime()


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


def calculate_regression_coefficient(x_series: pd.Series, y_series: pd.Series) -> float:
    slope, _ = np.polyfit(x_series.values, y_series.values, deg=1)
    return float(slope)


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


# def create_returns_from_price_history(price_history_df: pd.DataFrame) -> float:
#     pair_price_returns_df = price_history_df.pct_change() * 100
#     return pair_price_returns_df


def are_stock_betas_similar(
    ticker_one: str, ticker_two: str, ticker_to_beta_map: Mapping[str, float]
) -> bool:
    beta_absolute_difference_threshold = 0.02
    relative_tolerance = 0
    are_betas_similar = np.isclose(
        ticker_to_beta_map[ticker_one],
        ticker_to_beta_map[ticker_two],
        atol=beta_absolute_difference_threshold,
        rtol=relative_tolerance,
    )
    return bool(are_betas_similar)


def get_stock_and_benchmark_price_history_df_algined_on_date(
    ticker_one: str,
    ticker_two: str,
    benchmark_ticker: str,
    start_date: datetime,
    end_date: datetime,
    all_tickers_price_history_dict: Mapping[str, pd.DataFrame],
    benhcmark_price_history_dict: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    ticker_one_price_history_series = filter_price_history_series_or_df_by_date(
        start_date, end_date, all_tickers_price_history_dict[ticker_one]
    )
    ticker_two_price_history_series = filter_price_history_series_or_df_by_date(
        start_date, end_date, all_tickers_price_history_dict[ticker_two]
    )
    benchmark_price_history_series = filter_price_history_series_or_df_by_date(
        start_date, end_date, benhcmark_price_history_dict[benchmark_ticker]
    )
    return pd.concat(
        [
            ticker_one_price_history_series,
            ticker_two_price_history_series,
            benchmark_price_history_series,
        ],
        axis=1,
    ).dropna()


# TODO this is basically identical to get_stock_and_benchmark_price_history_df_algined_on_date
# def get_pair_price_history_df_algined_on_date(
#     ticker_one: str,
#     ticker_two: str,
#     start_date: datetime,
#     end_date: datetime,
#     all_tickers_price_history_dict: Mapping[str, pd.DataFrame],
# ) -> pd.DataFrame:
#     ticker_one_price_history_series = filter_price_history_series_or_df_by_date(
#         start_date, end_date, all_tickers_price_history_dict[ticker_one]
#     )
#     ticker_two_price_history_series = filter_price_history_series_or_df_by_date(
#         start_date, end_date, all_tickers_price_history_dict[ticker_two]
#     )
#     return pd.concat(
#         [
#             ticker_one_price_history_series,
#             ticker_two_price_history_series,
#         ],
#         axis=1,
#     ).dropna()


# def filter_price_history_series_or_df_by_date(
#     start_date: datetime,
#     end_date: datetime,
#     stock_and_benchmark_price_history_df: Union[pd.DataFrame, pd.Series],
# ) -> Union[pd.DataFrame, pd.Series]:
#     return stock_and_benchmark_price_history_df[
#         (stock_and_benchmark_price_history_df.index >= start_date)
#         & (stock_and_benchmark_price_history_df.index <= end_date)
#     ]


def are_tickers_in_same_sector(
    ticker_one: str, ticker_two: str, ticker_to_sector: Mapping[str, str]
) -> bool:
    return ticker_to_sector[ticker_one] == ticker_to_sector[ticker_two]


def create_ticker_to_sector_map() -> Mapping[str, str]:
    with open("data/ticker_to_sector.pkl", "rb") as ticker_to_sector_map_file:
        return pickle.load(ticker_to_sector_map_file)


def get_ticker_price_series_filtered_by_date(
    start_date_for_cointegration_period: datetime,
    end_date_for_cointegration_period: datetime,
    ticker_one: str,
    ticker_two: str,
    all_tickers_price_history_df: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    all_tickers_price_history_filtered_by_date_df = (
        filter_price_history_series_or_df_by_date(
            start_date_for_cointegration_period,
            end_date_for_cointegration_period,
            all_tickers_price_history_df,
        )
    )
    return (
        all_tickers_price_history_filtered_by_date_df[ticker_one],
        all_tickers_price_history_filtered_by_date_df[ticker_two],
    )


# NOTE: This methodology (ex beta filter) came from Caldeira & Caldeira 2013. Paper is in references folder
def is_pair_cointegrated(
    ticker_one_price_series: pd.Series,
    ticker_two_price_series: pd.Series,
) -> bool:
    try:
        if not is_price_series_integrated_of_order_one(ticker_one_price_series):
            return False
        if not is_price_series_integrated_of_order_one(ticker_two_price_series):
            return False
        if not is_pair_engle_granger_cointegrated(
            ticker_one_price_series, ticker_two_price_series
        ):
            return False
        if not is_pair_johansen_cointegrated(
            ticker_one_price_series, ticker_two_price_series
        ):
            return False
        return True
    except Exception as e:
        print(
            f"Unable to evaluate pair {ticker_one_price_series.name + '/' + ticker_two_price_series.name}"
        )
        print(e)
        return False


# def calculate_gamma(
#     pair_price_history_df: pd.DataFrame,
#     in_sample_start_date: datetime,
#     in_sample_end_date: datetime,
# ) -> float:
#     date_mask = (pair_price_history_df.index >= in_sample_start_date) & (
#         pair_price_history_df.index <= in_sample_end_date
#     )
#     pair_price_history_in_sample_df = pair_price_history_df[date_mask]
#     ticker1_price_series = pair_price_history_in_sample_df.iloc[:, 0]
#     ticker2_price_series = pair_price_history_in_sample_df.iloc[:, 1]
#     slope, _ = np.polyfit(
#         ticker2_price_series.values, ticker1_price_series.values, deg=1
#     )
#     return float(slope)


# def calculate_spread(
#     pair_price_history_df: pd.DataFrame,
#     gamma: float,
#     start_date: datetime,
#     end_date: datetime,
# ) -> pd.Series:
#     date_mask = (pair_price_history_df.index >= start_date) & (
#         pair_price_history_df.index <= end_date
#     )
#     pair_price_history_out_of_sample_df = pair_price_history_df[date_mask]
#     ticker1_price_series = pair_price_history_out_of_sample_df.iloc[:, 0]
#     ticker2_price_series = pair_price_history_out_of_sample_df.iloc[:, 1]
#     return ticker1_price_series - gamma * ticker2_price_series


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


# def get_ticker_list() -> list[tuple[str, str]]:
#     path_to_ticker_list = Path("data/russell_3000_constituents.csv")
#     ticker_df = pd.read_csv(path_to_ticker_list)
#     return ticker_df[_TICKER_COLUMN_NAME].to_list()


def get_all_possible_ticker_pairs(
    all_tickers_price_history_df: pd.DataFrame,
) -> list[tuple]:
    return list(combinations(all_tickers_price_history_df.columns, 2))


# def get_all_tickers_price_history_df(
#     start_date: datetime, end_date: datetime
# ) -> pd.DataFrame:
#     all_tickers_price_history = []
#     for ticker in get_ticker_list():
#         path_to_ticker_price_history_csv = (
#             Path("data") / "adj_close_price_data" / f"{ticker}.csv"
#         )
#         try:
#             all_tickers_price_history.append(
#                 pd.read_csv(
#                     path_to_ticker_price_history_csv, index_col=0, parse_dates=True
#                 )
#             )
#         except FileNotFoundError:
#             print(f"No data found for ticker: {ticker}")
#             continue
#     all_tickers_price_history_df = pd.concat(all_tickers_price_history, axis=1)
#     return filter_price_history_series_or_df_by_date(
#         start_date, end_date, all_tickers_price_history_df
#     )


def get_benchmark_price_history_df(
    start_date: datetime, end_date: datetime, benchmark_ticker: str
) -> pd.DataFrame:
    path_to_benchmark_price_history_data = Path(
        f"data/adj_close_price_data_benchmarks/{benchmark_ticker}.csv"
    )
    benchmark_price_history_df = pd.read_csv(
        path_to_benchmark_price_history_data, index_col=0, parse_dates=True
    )
    return filter_price_history_series_or_df_by_date(
        start_date, end_date, benchmark_price_history_df
    )


# TODO this probably won't need to write to disk once full walk-forward model is implemented
def get_cointegrated_pairs(
    start_date_for_cointegration_period: datetime,
    end_date_for_cointegration_period: datetime,
    tickers_price_history_df: pd.DataFrame,
    pairs_with_common_sector_and_beta: Collection[tuple[str, str]],
) -> Sequence[tuple[str, str]]:
    valid_pairs = []
    filtered_tickers_price_history_df = filter_price_history_series_or_df_by_date(
        start_date_for_cointegration_period,
        end_date_for_cointegration_period,
        tickers_price_history_df,
    )
    for i, (ticker_one, ticker_two) in enumerate(pairs_with_common_sector_and_beta):
        print(i)
        if is_pair_cointegrated(
            filtered_tickers_price_history_df[ticker_one],
            filtered_tickers_price_history_df[ticker_two],
        ):
            valid_pairs.append((ticker_one, ticker_two))
            print("LENGTH VALID PAIRS: ", len(valid_pairs))

    valid_pairs_output_path = Path(f"data/valid_pairs_{datetime.now()}.csv")
    pd.DataFrame({"Valid Pairs": valid_pairs}).to_csv(
        valid_pairs_output_path,
        mode="a",
        header=False,
        index=False,
    )
    return valid_pairs


def calculate_hust_exponent_for_pairs(
    start_date_for_cointegration_period: datetime,
    end_date_for_cointegration_period: datetime,
    cointegrated_pairs: Collection[tuple[str, str]],
    cointegrated_ticker_price_history_df: pd.DataFrame,
) -> Collection[tuple[str, str, float]]:
    pair_and_husrt_exponents = []
    for ticker_one, ticker_two in cointegrated_pairs:
        pair_price_history_df = get_pair_price_history_df_algined_on_date(
            ticker_one,
            ticker_two,
            start_date_for_cointegration_period,
            end_date_for_cointegration_period,
            cointegrated_ticker_price_history_df,
        )
        gamma = calculate_gamma(
            pair_price_history_df,
            start_date_for_cointegration_period,
            end_date_for_cointegration_period,
        )
        spread_series = calculate_spread(
            pair_price_history_df,
            gamma,
            start_date_for_cointegration_period,
            end_date_for_cointegration_period,
        )
        pair_and_husrt_exponents.append(
            (ticker_one, ticker_two, calculate_generalized_hurst_exponent_q1(spread_series))
        )
    return pair_and_husrt_exponents


def filter_price_history_df_by_pairs(
    cointegrated_pairs: Collection[tuple[str, str]],
    all_tickers_price_history_df: pd.DataFrame,
) -> pd.DataFrame:
    tickers_in_cointegrated_pair = set()
    for ticker_one, ticker_two in cointegrated_pairs:
        tickers_in_cointegrated_pair.add(ticker_one)
        tickers_in_cointegrated_pair.add(ticker_two)
    return all_tickers_price_history_df[list(tickers_in_cointegrated_pair)]


# NOTE For speed reasons all data required by this must be written to
# target folders beforehand


# TODO benchmark constituents are currently not PIT so there is technically survivorship bias
def inner_join_series_on_date_index(
    series_one: pd.Series, series_two: pd.Series
) -> pd.DataFrame:
    return pd.concat([series_one, series_two], axis=1, join="inner").dropna()


def create_ticker_to_beta_map(
    benchmark_ticker: str,
    all_tickers_and_benchmark_price_history_df: pd.DataFrame,
) -> Mapping[str, float]:
    ticker_to_beta_map = {}
    ticker_and_benchmark_returns_df = create_returns_from_price_history(
        all_tickers_and_benchmark_price_history_df
    )
    for ticker in ticker_and_benchmark_returns_df.columns:
        if ticker == benchmark_ticker:
            continue
        if ticker_and_benchmark_returns_df[ticker].count() < _ONE_YEAR_IN_TRADING_DAYS:
            print("No price data over desired period for: ", ticker)
            continue
        benchmark_and_ticker_returns_df = inner_join_series_on_date_index(
            ticker_and_benchmark_returns_df[benchmark_ticker],
            ticker_and_benchmark_returns_df[ticker],
        )
        ticker_to_beta_map[ticker] = calculate_regression_coefficient(
            benchmark_and_ticker_returns_df[benchmark_ticker],
            benchmark_and_ticker_returns_df[ticker],
        )
    return ticker_to_beta_map


def get_pairs_with_common_beta(
    pairs: Collection[tuple[str, str]],
    all_tickers_price_history_df: pd.DataFrame,
) -> Collection[tuple[str, str]]:
    benchmark_ticker = "IWV"
    pairs_with_common_beta = []
    benchmark_price_history_df = get_benchmark_price_history_df(
        start_date_for_beta_estimation_period,
        end_date_for_cointegration_period,
        benchmark_ticker,
    )
    all_tickers_and_benchmark_price_history_df = (
        join_ticker_and_benchmark_price_history_dfs(
            all_tickers_price_history_df, benchmark_price_history_df
        )
    )
    ticker_to_beta_map = create_ticker_to_beta_map(
        benchmark_ticker, all_tickers_and_benchmark_price_history_df
    )
    for ticker_one, ticker_two in pairs:
        if ticker_one not in ticker_to_beta_map or ticker_two not in ticker_to_beta_map:
            continue
        if are_stock_betas_similar(
            ticker_one,
            ticker_two,
            ticker_to_beta_map,
        ):
            pairs_with_common_beta.append((ticker_one, ticker_two))
    return pairs_with_common_beta


def get_pairs_with_common_sector(
    pairs: Collection[tuple[str, str]],
) -> Collection[tuple[str, str]]:
    pairs_with_common_sector = []
    ticker_to_sector_map = create_ticker_to_sector_map()
    for ticker_one, ticker_two in pairs:
        if are_tickers_in_same_sector(ticker_one, ticker_two, ticker_to_sector_map):
            pairs_with_common_sector.append((ticker_one, ticker_two))
    return pairs_with_common_sector


def join_ticker_and_benchmark_price_history_dfs(
    all_tickers_price_history_df: pd.DataFrame,
    benchmark_price_history_df: pd.DataFrame,
) -> pd.DataFrame:
    return pd.concat([all_tickers_price_history_df, benchmark_price_history_df], axis=1)


def get_all_tickers_and_benchmark_price_history_df(
    start_date_for_beta_estimation_period: datetime,
    end_date_for_cointegration_period: datetime,
    benchmark_ticker: str,
) -> pd.DataFrame:
    all_tickers_price_history_df = get_all_tickers_price_history_df(
        start_date_for_beta_estimation_period, end_date_for_cointegration_period
    )
    benchmark_price_history_df = get_benchmark_price_history_df(
        start_date_for_beta_estimation_period,
        end_date_for_cointegration_period,
        benchmark_ticker,
    )
    return join_ticker_and_benchmark_price_history_dfs(
        all_tickers_price_history_df, benchmark_price_history_df
    )

def get_cointegrated_pairs_with_hurst_exponent(
    end_date_for_cointegration_test_period: datetime,
    beta_estimation_period_in_trading_days: int,
    cointegration_test_period_in_trading_days: int
) -> Collection[tuple[str, str, float]]:
    start_date_for_cointegration_test_period = subtract_n_us_trading_days_from_date(
        end_date_for_cointegration_test_period, cointegration_test_period_in_trading_days
    )
    start_date_for_beta_estimation_period = subtract_n_us_trading_days_from_date(
        end_date_for_cointegration_test_period, beta_estimation_period_in_trading_days
    )
    all_tickers_price_history_df = get_all_tickers_price_history_df(
        start_date_for_beta_estimation_period, end_date_for_cointegration_test_period
    )
    all_possible_pairs = get_all_possible_ticker_pairs(all_tickers_price_history_df)
    pairs_with_common_sector = get_pairs_with_common_sector(all_possible_pairs)
    pairs_with_common_sector_and_beta = get_pairs_with_common_beta(
        pairs_with_common_sector, all_tickers_price_history_df
    )
    ticker_price_history_filtered_by_sector_and_beta_df = filter_price_history_df_by_pairs(
        pairs_with_common_sector_and_beta, all_tickers_price_history_df
    )
    cointegrated_pairs = get_cointegrated_pairs(
        start_date_for_cointegration_test_period,
        end_date_for_cointegration_test_period,
        ticker_price_history_filtered_by_sector_and_beta_df,
        pairs_with_common_sector_and_beta,
    )
    cointegrated_ticker_price_history_df = filter_price_history_df_by_pairs(
        cointegrated_pairs, ticker_price_history_filtered_by_sector_and_beta_df
    )
    return calculate_hust_exponent_for_pairs(
        start_date_for_cointegration_test_period,
        end_date_for_cointegration_test_period,
        cointegrated_pairs,
        cointegrated_ticker_price_history_df,
    )

# if __name__ == "__main__":
#     end_date_for_cointegration_period = datetime(2023, 12, 31)
#     start_date_for_cointegration_period = subtract_n_us_trading_days_from_date(
#         end_date_for_cointegration_period, _ONE_YEAR_IN_TRADING_DAYS
#     )
#     start_date_for_beta_estimation_period = subtract_n_us_trading_days_from_date(
#         end_date_for_cointegration_period, _BETA_ESTIMATION_PERIOD_IN_TRADING_DAYS
#     )
#     all_tickers_price_history_df = get_all_tickers_price_history_df(
#         start_date_for_beta_estimation_period, end_date_for_cointegration_period
#     )
#     all_possible_pairs = get_all_possible_ticker_pairs(all_tickers_price_history_df)
#     pairs_with_common_sector = get_pairs_with_common_sector(all_possible_pairs)
#     pairs_with_common_sector_and_beta = get_pairs_with_common_beta(
#         pairs_with_common_sector, all_tickers_price_history_df
#     )
#     ticker_price_history_filtered_by_sector_and_beta_df = filter_price_history_df_by_pairs(
#         pairs_with_common_sector_and_beta, all_tickers_price_history_df
#     )
#     cointegrated_pairs = get_cointegrated_pairs(
#         start_date_for_cointegration_period,
#         end_date_for_cointegration_period,
#         ticker_price_history_filtered_by_sector_and_beta_df,
#         pairs_with_common_sector_and_beta,
#     )
#     cointegrated_ticker_price_history_df = filter_price_history_df_by_pairs(
#         cointegrated_pairs, ticker_price_history_filtered_by_sector_and_beta_df
#     )
#     hurst_components_of_pair_spreads = calculate_hust_exponent_for_pairs(
#         start_date_for_cointegration_period,
#         end_date_for_cointegration_period,
#         cointegrated_pairs,
#         cointegrated_ticker_price_history_df,
#     )
#     breakpoint()

# NOTE for the period ending at (2023, 12, 31):
# 975,562 pairs are 1) in the same sector and 2) have a beta within .3
# 665,837 pairs are 1) in the same sector and 2) have a beta within .2
# 503,467 pairs are 1) in the same sector and 2) have a beta within .15
# 338,030 pairs are 1) in the same sector and 2) have a beta within .1
# 169,816 pairs are 1) in the same sector and 2) have a beta within .05
# 68,036 pairs are 1) in the same sector and 2) have a beta within .02

# This is an example of one strategy to reduce the time to evaluate pairs.
# Another option is parrallelizing the program. The value of the latter
# depends on the quantity and quality of the cointegrated pairs that the stricter
# beta limit produces.
