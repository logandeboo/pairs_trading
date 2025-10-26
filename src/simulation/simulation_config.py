from src.universe import Universe
from datetime import datetime
from typing import NamedTuple


class SimulationConfig(NamedTuple):
    universe: Universe
    start_date: datetime
    end_date: datetime
    rebalance_freq_in_trading_days: int
    cointegration_test_period_in_trading_days: int
    num_pairs_in_portfolio: int
