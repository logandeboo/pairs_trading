from datetime import datetime
from src.pair_selection import get_pairs_for_portfolio_simulation
from src.time_series_utils import (
    subtract_n_us_trading_days_from_date,
    ONE_YEAR_IN_TRADING_DAYS,
    ONE_DAY_IN_TRADING_DAYS,
)
from src.data_loader import get_all_tickers_price_history_df
from src.simulation.simulation import get_simulated_returns_by_portfolio_tickers_df
from src.performance import get_portfolio_performance_result


SIMULATION_PERIOD_IN_TRADING_DAYS = ONE_YEAR_IN_TRADING_DAYS
COINTEGRATION_TEST_PERIOD_IN_TRADING_DAYS = ONE_YEAR_IN_TRADING_DAYS
BETA_ESTIMATION_PERIOD_IN_TRADING_DAYS = ONE_YEAR_IN_TRADING_DAYS * 2
NUM_PAIRS_IN_PORTFOLIO = 10
SPREAD_Z_SCORE_ROLLING_WINDOW_IN_TRADING_DAYS = 20

import pandas as pd
import ast


def parse_pair_df():
    df = pd.read_csv(
        "/Users/LoganDeboo/Desktop/projects/pairs_trading/scratchwork/hurst_exponents.csv",
        index_col=0,
    )
    parsed_list = []
    for _, row in df.iterrows():
        tickers = row[0]
        value = row[1]
        # Convert string representation to actual tuple if needed
        if isinstance(tickers, str):
            tickers = ast.literal_eval(tickers)
        if isinstance(tickers, tuple) and len(tickers) == 2:
            parsed_list.append((str(tickers[0]), str(tickers[1])))
        else:
            raise ValueError(f"Unexpected format in row: {tickers}")
    return parsed_list


if __name__ == "__main__":
    simulation_end_date = datetime(2023, 1, 1)
    simulation_start_date = subtract_n_us_trading_days_from_date(
        simulation_end_date, offset_in_us_trading_days=SIMULATION_PERIOD_IN_TRADING_DAYS
    )
    cointegration_test_end_date = subtract_n_us_trading_days_from_date(
        simulation_start_date, offset_in_us_trading_days=ONE_DAY_IN_TRADING_DAYS
    )
    cointegration_test_start_date = subtract_n_us_trading_days_from_date(
        cointegration_test_end_date,
        offset_in_us_trading_days=COINTEGRATION_TEST_PERIOD_IN_TRADING_DAYS,
    )
    beta_calculation_start_date = subtract_n_us_trading_days_from_date(
        simulation_start_date,
        offset_in_us_trading_days=BETA_ESTIMATION_PERIOD_IN_TRADING_DAYS,
    )
    all_tickers_price_history_df = get_all_tickers_price_history_df(
        beta_calculation_start_date, cointegration_test_end_date
    )
    pairs_for_portfolio_simulation = get_pairs_for_portfolio_simulation(
        cointegration_test_start_date,
        cointegration_test_end_date,
        all_tickers_price_history_df,
        num_pairs_in_portfolio=NUM_PAIRS_IN_PORTFOLIO,
    )
    pairs_for_portfolio_simulation = parse_pair_df()
    simulation_returns_by_ticker_df = get_simulated_returns_by_portfolio_tickers_df(
        simulation_start_date,
        simulation_end_date,
        pairs_for_portfolio_simulation,
        all_tickers_price_history_df,
        spread_z_score_rolling_window_in_trading_days=SPREAD_Z_SCORE_ROLLING_WINDOW_IN_TRADING_DAYS,
    )
    # portfolio_performance_result = get_portfolio_performance_result(
    #     simulation_returns_by_ticker_df
    # )


# TODO interface of strategy_pipeline should be two functions:
# get_pairs_for_portfolio_simulation and get_portfolio_performance_result
