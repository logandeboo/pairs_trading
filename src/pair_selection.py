import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from itertools import combinations
from typing import Sequence, Mapping, Union, Collection

from src.time_series_utils import (
    filter_price_history_series_or_df_by_date_inclusive,
    filter_price_history_df_by_pair_and_date,
    ONE_YEAR_IN_TRADING_DAYS,
)
from src.data_loader import (
    get_all_tickers_price_history_df,
    create_ticker_to_sector_map,
    get_benchmark_price_history_df,
)
from src.statistical_utils import (
    calculate_regression_coefficient,
    get_pair_spread_series,
    calculate_generalized_hurst_exponent_q1,
    is_price_series_integrated_of_order_one,
    is_pair_engle_granger_cointegrated,
    is_pair_johansen_cointegrated,
    create_returns_from_price_history,
)


def are_stock_betas_similar(
    ticker_one: str, ticker_two: str, ticker_to_beta_map: Mapping[str, float]
) -> bool:
    beta_absolute_difference_threshold = 0.02
    relative_tolerance = 0
    if ticker_one not in ticker_to_beta_map or ticker_two not in ticker_to_beta_map:
        return False
    are_betas_similar = np.isclose(
        ticker_to_beta_map[ticker_one],
        ticker_to_beta_map[ticker_two],
        atol=beta_absolute_difference_threshold,
        rtol=relative_tolerance,
    )
    return bool(are_betas_similar)


def are_tickers_in_same_sector(
    ticker_one: str, ticker_two: str, ticker_to_sector: Mapping[str, str]
) -> bool:
    return ticker_to_sector[ticker_one] == ticker_to_sector[ticker_two]


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


def create_ticker_pairs(
    all_tickers_price_history_df: pd.DataFrame,
) -> list[tuple]:
    return list(combinations(all_tickers_price_history_df.columns, 2))


# TODO this probably won't need to write to disk once full walk-forward model is implemented
def filter_pairs_for_cointegration(
    start_date_for_cointegration_period: datetime,
    end_date_for_cointegration_period: datetime,
    pairs_with_common_sector_and_beta: Collection[tuple[str, str]],
    tickers_price_history_df: pd.DataFrame,
) -> Sequence[tuple[str, str]]:
    valid_pairs = []
    filtered_tickers_price_history_df = (
        filter_price_history_series_or_df_by_date_inclusive(
            start_date_for_cointegration_period,
            end_date_for_cointegration_period,
            tickers_price_history_df,
        )
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
        pair_price_history_df = filter_price_history_df_by_pair_and_date(
            ticker_one,
            ticker_two,
            start_date_for_cointegration_period,
            end_date_for_cointegration_period,
            cointegrated_ticker_price_history_df,
        )
        spread_series = get_pair_spread_series(
            pair_price_history_df,
        )
        pair_and_husrt_exponents.append(
            (
                ticker_one,
                ticker_two,
                calculate_generalized_hurst_exponent_q1(spread_series),
            )
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


# TODO benchmark constituents are currently not PIT so there is survivorship bias
def inner_join_series_on_date_index(
    series_one: pd.Series, series_two: pd.Series
) -> pd.DataFrame:
    return pd.concat([series_one, series_two], axis=1, join="inner").dropna()


def has_enough_returns_to_estimate_beta(
    ticker: str,
    benchmark_and_ticker_returns_df: pd.DataFrame,
) -> bool:
    if len(benchmark_and_ticker_returns_df) < ONE_YEAR_IN_TRADING_DAYS:
        print(f"Returns history not long enough to calculate beta for {ticker}")


def create_ticker_to_beta_map(
    start_date_for_beta_estimation_period: datetime,
    end_date_for_cointegration_period: datetime,
    all_tickers_price_history_df: pd.DataFrame,
) -> Mapping[str, float]:
    benchmark_ticker = "IWV"
    ticker_to_beta_map = {}
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
    ticker_and_benchmark_returns_df = create_returns_from_price_history(
        all_tickers_and_benchmark_price_history_df
    )
    for ticker in all_tickers_price_history_df.columns:
        benchmark_and_ticker_returns_df = ticker_and_benchmark_returns_df[
            [ticker, benchmark_ticker]
        ].dropna()
        if not has_enough_returns_to_estimate_beta(
            ticker, benchmark_and_ticker_returns_df
        ):
            continue
        ticker_to_beta_map[ticker] = calculate_regression_coefficient(
            benchmark_and_ticker_returns_df[benchmark_ticker],
            benchmark_and_ticker_returns_df[ticker],
        )
    return ticker_to_beta_map


def filter_pairs_for_common_beta(
    start_date_for_beta_estimation_period: datetime,
    end_date_for_cointegration_period: datetime,
    pairs: Collection[tuple[str, str]],
    all_tickers_price_history_df: pd.DataFrame,
) -> Collection[tuple[str, str]]:
    pairs_with_common_beta = []
    ticker_to_beta_map = create_ticker_to_beta_map(
        start_date_for_beta_estimation_period,
        end_date_for_cointegration_period,
        all_tickers_price_history_df,
    )
    for ticker_one, ticker_two in pairs:
        if are_stock_betas_similar(
            ticker_one,
            ticker_two,
            ticker_to_beta_map,
        ):
            pairs_with_common_beta.append((ticker_one, ticker_two))
    return pairs_with_common_beta


def filter_pairs_for_common_sector(
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


def filter_cointegrated_pairs_by_hurst_exponent(
    cointegrated_pairs_with_hurst_exponent: Collection[tuple[str, str]],
    num_pairs_in_portfolio: int,
) -> Collection[tuple[str, str, float]]:
    pairs_sorted_by_hurst_exponent = [
        (ticker_one, ticker_two)
        for ticker_one, ticker_two, _ in sorted(
            cointegrated_pairs_with_hurst_exponent, key=lambda x: x[2], reverse=False
        )
    ]
    return pairs_sorted_by_hurst_exponent[: num_pairs_in_portfolio + 1]


def get_pairs_for_portfolio_simulation(
    start_date_for_cointegration_test_period: datetime,
    end_date_for_cointegration_test_period: datetime,
    all_tickers_price_history_df: pd.DataFrame,
    *,
    num_pairs_in_portfolio: int,
) -> Collection[tuple[str, str]]:
    all_possible_pairs = create_ticker_pairs(all_tickers_price_history_df)
    pairs_with_common_sector = filter_pairs_for_common_sector(all_possible_pairs)
    pairs_with_common_sector_and_beta = filter_pairs_for_common_beta(
        start_date_for_cointegration_test_period,
        end_date_for_cointegration_test_period,
        pairs_with_common_sector,
        all_tickers_price_history_df,
    )
    cointegrated_pairs = filter_pairs_for_cointegration(
        start_date_for_cointegration_test_period,
        end_date_for_cointegration_test_period,
        pairs_with_common_sector_and_beta,
        all_tickers_price_history_df,
    )
    cointegrated_pairs_with_hurst_exponent = calculate_hust_exponent_for_pairs(
        start_date_for_cointegration_test_period,
        end_date_for_cointegration_test_period,
        cointegrated_pairs,
        all_tickers_price_history_df,
    )
    return filter_cointegrated_pairs_by_hurst_exponent(
        cointegrated_pairs_with_hurst_exponent, num_pairs_in_portfolio
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
