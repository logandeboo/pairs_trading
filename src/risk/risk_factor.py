from enum import Enum
import pandas as pd


def get_risk_factor_returns_df(name: str) -> pd.DataFrame:
    raise NotImplementedError


class RiskFactor:
    def __init__(
        self,
        name: str,
    ) -> None:
        self.name = name
        self.returns_df = get_risk_factor_returns_df(name)


class RiskFactors(Enum):
    HIGH_MINUS_LOW = RiskFactor("high_minus_low_fama_french")
    SMALL_MINUS_BIG = RiskFactor("small_minus_big_fama_french")
