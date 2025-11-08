from typing import NamedTuple, Mapping
import pandas as pd
from src.risk.risk_factor import RiskFactor

class Stock(NamedTuple):
    ticker: str
    sector: str
    daily_price_history_df: pd.DataFrame
    daily_returns_df: pd.DataFrame

    def __eq__(self, other):
        if not isinstance(other, Stock):
            return NotImplemented
        return self.ticker == other.ticker
    
    def __hash__(self):
        return hash(self.ticker)



