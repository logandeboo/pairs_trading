from datetime import datetime
from typing import NamedTuple


# TODO implement calculations for commented out attributes
class SimulationResult(NamedTuple):
    annualized_return: float
    annualized_volatility: float
    annualized_sharpe_ratio: float
    # hit_rate: float
    # max_drawdown: float
    # max_drawdown_duration_in_trading_days: float
    # skewness: float
    performance_period_start: datetime
    performance_period_end: datetime
