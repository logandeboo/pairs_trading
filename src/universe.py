from typing import Collection
from src.risk.risk_factor import RiskFactor
from src.stock import Stock
from typing import Mapping
from enum import Enum
from pathlib import Path
import pandas as pd
from src.stock import get_all_stocks_in_universe
import pickle

class UniverseName(Enum):
    USA = "USA"


# TODO this should be replaced with a largest set of US tickers
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

def get_universe_tickers(universe_name: UniverseName) -> list[tuple[str, str]]:
    path_to_ticker_list = _UNIVERSE_NAME_TO_CONSITUENTS_FILE_PATH[universe_name]
    ticker_df = pd.read_csv(path_to_ticker_list)
    return ticker_df.squeeze().to_list()


class Universe:

    def __init__(
        self,
        name: UniverseName,
        risk_factor_to_similarity_constraint: Mapping[RiskFactor, float],
    ) -> None:
        self.name = name
        self.stocks = get_all_stocks_in_universe(
            name,
        )
        self.risk_factor_to_similarity_constraint = risk_factor_to_similarity_constraint
