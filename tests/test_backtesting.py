from src.backtesting import calculate_return_from_equal_dollar_weight_trades
import pandas as pd
import numpy as np


class TestCalculateReturnFromEqualDollarWeighTrades:

    def test_empty_series(self) -> None:
        expected = 0
        returns = pd.Series()
        actual = calculate_return_from_equal_dollar_weight_trades(returns)
        assert expected == actual

    def test_series_size_one(self) -> None:
        expected = 0.56
        returns = pd.Series([expected])
        actual = calculate_return_from_equal_dollar_weight_trades(returns)
        assert expected == actual

    def test_leading_zero(self) -> None:
        returns = [0.1, 0.1]
        returns_no_leading_zero = pd.Series(returns)
        returns_with_leading_zero = pd.Series([0] + returns)
        assert calculate_return_from_equal_dollar_weight_trades(
            returns_no_leading_zero
        ) == calculate_return_from_equal_dollar_weight_trades(returns_with_leading_zero)

    def test_trailing_zero(self) -> None:
        returns = [0.2, 0.2]
        returns_no_trailing_zero = pd.Series(returns)
        returns_with_trailing_zero = pd.Series(returns + [0])
        assert calculate_return_from_equal_dollar_weight_trades(
            returns_no_trailing_zero
        ) == calculate_return_from_equal_dollar_weight_trades(
            returns_with_trailing_zero
        )

    def test_toy_series(self) -> None:
        returns = pd.Series(
            [0.1, 0.1, 0.1, 0, 0, 0.12, -0.34, -0.24]
        )
        expected = -0.10720799999999964
        actual = calculate_return_from_equal_dollar_weight_trades(returns)
        assert expected == actual

    def test_sample_data(self) -> None:
        expected = 0.12098087375052535
        returns = pd.read_csv(
            "tests/test_data/KVUE_trade_returns.csv", index_col=0, parse_dates=True
        )["KVUE"]
        actual = calculate_return_from_equal_dollar_weight_trades(returns)
        assert np.isclose(expected, actual)