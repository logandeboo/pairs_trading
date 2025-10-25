from datetime import datetime
from src.pair_selection import get_pairs_for_portfolio_simulation
from src.time_series_utils import (
    subtract_n_us_trading_days_from_date,
    ONE_YEAR_IN_TRADING_DAYS,
    ONE_DAY_IN_TRADING_DAYS,
)
from src.data_loader import get_all_tickers_price_history_df
from src.simulation import get_all_pair_returns_by_ticker_df
from src.performance import get_portfolio_performance_result

SPREAD_Z_SCORE_ROLLING_WINDOW_IN_TRADING_DAYS = 20
NUM_PAIRS_IN_PORTFOLIO = 10

SIMULATION_PERIOD_IN_TRADING_DAYS = ONE_YEAR_IN_TRADING_DAYS
COINTEGRATION_TEST_PERIOD_IN_TRADING_DAYS = ONE_YEAR_IN_TRADING_DAYS
BETA_ESTIMATION_PERIOD_IN_TRADING_DAYS = ONE_YEAR_IN_TRADING_DAYS * 2

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
returns_by_ticker_df = get_all_pair_returns_by_ticker_df(
    simulation_start_date,
    simulation_end_date,
    pairs_for_portfolio_simulation,
    spread_z_score_rolling_window_in_trading_days=SPREAD_Z_SCORE_ROLLING_WINDOW_IN_TRADING_DAYS,
)
portfolio_performance_result = get_portfolio_performance_result(returns_by_ticker_df)
