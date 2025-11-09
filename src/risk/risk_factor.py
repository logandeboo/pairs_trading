import pandas as pd
from pathlib import Path


def get_path_to_risk_factor_returns_df(risk_factor_name: str) -> Path:
    path_to_risk_factor_dir = Path("data/risk_factor_returns")
    return path_to_risk_factor_dir / f"{risk_factor_name}.csv"


def get_risk_factor_returns_df(risk_factor_name: str) -> pd.DataFrame:
    path_to_risk_factor_returns_df = get_path_to_risk_factor_returns_df(
        risk_factor_name
    )
    return pd.read_csv(path_to_risk_factor_returns_df, index_col=0, parse_dates=True)


class RiskFactor:
    def __init__(
        self,
        name: str,
    ) -> None:
        self.name = name
        self.returns_df = get_risk_factor_returns_df(name)
