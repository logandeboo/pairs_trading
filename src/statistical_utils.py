import pandas as pd
from datetime import datetime
import numpy as np



def calculate_spread(
    pair_price_history_df: pd.DataFrame,
    gamma: float,
    start_date: datetime,
    end_date: datetime,
) -> pd.Series:
    date_mask = (pair_price_history_df.index >= start_date) & (
        pair_price_history_df.index <= end_date
    )
    pair_price_history_out_of_sample_df = pair_price_history_df[date_mask]
    ticker1_price_series = pair_price_history_out_of_sample_df.iloc[:, 0]
    ticker2_price_series = pair_price_history_out_of_sample_df.iloc[:, 1]
    return ticker1_price_series - gamma * ticker2_price_series

# TODO replace this polyfit usage with helper
def calculate_gamma(
    pair_price_history_df: pd.DataFrame,
    in_sample_start_date: datetime,
    in_sample_end_date: datetime,
) -> float:
    date_mask = (pair_price_history_df.index >= in_sample_start_date) & (
        pair_price_history_df.index <= in_sample_end_date
    )
    pair_price_history_in_sample_df = pair_price_history_df[date_mask]
    ticker1_price_series = pair_price_history_in_sample_df.iloc[:, 0]
    ticker2_price_series = pair_price_history_in_sample_df.iloc[:, 1]
    slope, _ = np.polyfit(
        ticker2_price_series.values, ticker1_price_series.values, deg=1
    )
    return float(slope)


def create_returns_from_price_history(price_history_df: pd.DataFrame) -> float:
    pair_price_returns_df = price_history_df.pct_change() * 100
    return pair_price_returns_df