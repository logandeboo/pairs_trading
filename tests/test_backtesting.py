from src.backtesting import calculate_number_of_trades_per_day
import pandas as pd


class TestCalculateNumberOfTradesPerDay:

    def test_calculate_num_trades_per_day_toy_data(self) -> None:
        input_df = pd.DataFrame(
            {
                "A": [0, 0, 1, 1, 1, 0, 0, 1, 1],
                "B": [0, 1, 0, 1, 0, 1, 0, 0, 1],
                "C": [0, 1, 1, 1, 0, 1, 1, 1, 0],
            }
        )
        actual_series = calculate_number_of_trades_per_day(input_df)
        expected_series = pd.Series([0, 3, 1, 3, 1, 3, 0, 2, 3])
        pd.testing.assert_series_equal(actual_series.astype("int64"), expected_series)

    def test_calculate_num_trades_per_day_sample_data(self) -> None:
        input_df = pd.read_csv(
            "tests/test_data/trade_returns_for_tickers.csv",
            index_col=0,
            parse_dates=True,
        )
        actual_series = calculate_number_of_trades_per_day(input_df)
        expected_series = pd.read_csv(
            "tests/test_data/num_trades_per_day_expected.csv",
            index_col=0,
            parse_dates=True,
        ).squeeze("columns")
        pd.testing.assert_series_equal(
            actual_series.astype("int64"), expected_series, check_names=False
        )


# TODO finish testing this
class TestCalculateReturnOnEmployedCaptial:

    def test_calculate_return_on_employed_capital_single_ticker(self) -> None:

        input_df = pd.DataFrame({"A": []})
