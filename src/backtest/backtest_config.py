from src.universe.universe import Universe
from datetime import datetime
from typing import NamedTuple, Mapping, Sequence
from src.risk.risk_factor import RiskFactor
from src.time_series_utils import get_rebalance_dates


class BacktestConfig(NamedTuple):
    universe: Universe
    start_date: datetime
    end_date: datetime
    risk_factor_to_similarity_threshold: Mapping[RiskFactor, float]
    risk_factor_exposure_period_in_us_trading_days: int
    rebalance_freq_in_us_trading_days: int
    cointegration_test_period_in_trading_days: int
    trade_entrance_threshold_spread_abs_std: int
    trade_exit_threshold_spread_abs_std: int
    num_pairs_in_portfolio: int
