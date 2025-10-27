from typing import NamedTuple
import pandas as pd




class Stock(NamedTuple):
    ticker: str
    sector: str
    daily_price_history_df: pd.DataFrame
    daily_returns_df: pd.DataFrame



