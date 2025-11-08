from typing import Mapping, Collection
from enum import Enum
from pathlib import Path
import pandas as pd
import pickle
from src.stock import Stock
from src.risk.risk_factor import RiskFactor
from src.data_loader import (
    get_daily_price_history_df,
)
from src.statistical_utils import create_daily_returns

class UniverseName(Enum):
    USA = "USA"

# TODO this should be replaced with a larger set of US tickers
# survivorship bias within the universe seems unavoidable without
# spending $$$ 
_UNIVERSE_NAME_TO_CONSITUENTS_FILE_PATH = {
    UniverseName.USA: Path("data/russell_3000_constituents.csv")
}

_UNIVERSE_NAME_TO_TICKER_TO_SECTOR_MAP_FILE_PATH = {
    UniverseName.USA: Path("data/ticker_to_sector_map_usa.pkl")
}

def get_universe_ticker_to_sector_map(universe_name: UniverseName) -> Mapping[str, str]:
    path_to_ticker_to_sector_map = _UNIVERSE_NAME_TO_TICKER_TO_SECTOR_MAP_FILE_PATH[universe_name]
    with open(path_to_ticker_to_sector_map, "rb") as ticker_to_sector_map_file:
        return pickle.load(ticker_to_sector_map_file)

def get_all_tickers_in_universe(universe_name: UniverseName) -> list[tuple[str, str]]:
    path_to_ticker_list = _UNIVERSE_NAME_TO_CONSITUENTS_FILE_PATH[universe_name]
    ticker_df = pd.read_csv(path_to_ticker_list)
    return ticker_df.squeeze().to_list()



def get_all_stocks_in_universe(universe_name: UniverseName) -> Collection[Stock]:
    ticker_to_sector_map = get_universe_ticker_to_sector_map(universe_name)
    universe_tickers = get_all_tickers_in_universe(universe_name)
    return [
        Stock(
            ticker=ticker,
            sector=ticker_to_sector_map[ticker],
            daily_price_history_df=(price_history_df := get_daily_price_history_df(ticker)),
            daily_returns_df=(daily_returns_df := create_daily_returns(price_history_df)),
        )
        for ticker in universe_tickers
    ]

class Universe:

    def __init__(
        self,
        name: UniverseName,
    ) -> None:
        self.name = name
        self.stocks = get_all_stocks_in_universe(
            name,
        )
        self.ticker_to_sector_map = get_universe_ticker_to_sector_map(name)