from datetime import datetime
from src.backtest.backtest_config import BacktestConfig
from src.backtest.backtest import Backtest
from src.universe.universes import USA_UNIVERSE
from src.risk.risk_factors import (
    MARKET_RISK_US,
    HIGH_MINUS_LOW_US,
    SMALL_MINUS_BIG_US,
    CONSERVATIVE_MINUS_AGGRESSIVE_US,
    ROBUST_MINUS_WEAK_US,
)
from src.time_series_utils import ONE_YEAR_IN_TRADING_DAYS
risk_factor_to_similarity_threshold = {
    MARKET_RISK_US : 0.2,
    HIGH_MINUS_LOW_US : 0.2,
    SMALL_MINUS_BIG_US : 0.2,
    CONSERVATIVE_MINUS_AGGRESSIVE_US : 0.2,
    ROBUST_MINUS_WEAK_US: 0.2,
}

backtest_config = BacktestConfig(
    universe=USA_UNIVERSE,
    start_date=datetime(2020,1,1),
    end_date=datetime(2024,1,1),
    risk_factor_to_similarity_threshold=risk_factor_to_similarity_threshold,
    risk_factor_exposure_period_in_us_trading_days=ONE_YEAR_IN_TRADING_DAYS,
    rebalance_freq_in_us_trading_days=ONE_YEAR_IN_TRADING_DAYS,
    cointegration_test_period_in_trading_days=ONE_YEAR_IN_TRADING_DAYS,
    trade_entrance_threshold_spread_abs_std=2,
    trade_exit_threshold_spread_abs_std=0.2,
    num_pairs_in_portfolio=10,
)

backtest = Backtest(backtest_config)
backtest.run()



