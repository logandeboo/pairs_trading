from typing import NamedTuple, Collection, Mapping
import pandas as pd
from pathlib import Path
import pickle
from src.universe import (get_universe_tickers, get_universe_ticker_to_sector_map, UniverseName)
from src.statistical_utils import create_returns_from_price_history


class Stock(NamedTuple):
    ticker: str
    sector: str
    price_history_df: pd.DataFrame
    returns_df: pd.DataFrame


def get_path_to_stock_price_history(ticker: str) -> Path:
    path_to_price_history_dir = Path("data/adj_close_price_data")
    return path_to_price_history_dir / f"{ticker}.csv"


def get_stock_price_history_df(ticker: str) -> pd.DataFrame:
    path_to_stock_price_history = get_path_to_stock_price_history(ticker)
    try:
        pd.read_csv(path_to_stock_price_history, index_col=0, parse_dates=True)
    except FileNotFoundError:
        print(f"No price data found for ticker: {ticker}")


def get_all_stocks_in_universe(universe_name: UniverseName) -> Collection[Stock]:
    ticker_to_sector_map = get_universe_ticker_to_sector_map(universe_name)
    universe_tickers = get_universe_tickers(universe_name)
    return [
        Stock(
            ticker=ticker,
            sector=ticker_to_sector_map[ticker],
            price_history_df=(price_history := get_stock_price_history_df(ticker)),
            returns_df=create_returns_from_price_history(price_history),
        )
        for ticker in universe_tickers
    ]
