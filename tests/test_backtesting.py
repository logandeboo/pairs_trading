import pandas as pd
from src.backtesting import (
    calculate_number_of_trades_per_day,
    calculate_daily_return_on_employed_capital,
    calculate_daily_portfolio_return_before_t_costs,
    PORTFOLIO_DAILY_RETURN_AFTER_T_COST_COLUMN_NAME,
)


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

    def test_calculate_daily_portfolio_return_before_t_costs(self) -> None:
        expected_df = pd.read_csv(
            "tests/test_data/portfolio_daily_return_before_t_cost.csv",
            index_col=0,
            parse_dates=True,
        )
        trade_returns_for_all_tickers_df = pd.read_csv(
            "tests/test_data/trade_returns_for_tickers_2.csv",
            index_col=0,
            parse_dates=True,
        )
        actual_df = calculate_daily_portfolio_return_before_t_costs(trade_returns_for_all_tickers_df)
        pd.testing.assert_frame_equal(expected_df, actual_df)


    def test_calculate_return_on_employed_capital_single_toy_column(self) -> None:
        one_way_transaction_cost_in_basis_points = 10
        input_df = pd.DataFrame({"A": [0.0, 0.01, 0, 0.04, 0.03]})
        actual_df = calculate_daily_return_on_employed_capital(
            input_df, one_way_transaction_cost_in_basis_points
        )
        expected_df = pd.DataFrame(
            {
                PORTFOLIO_DAILY_RETURN_AFTER_T_COST_COLUMN_NAME: [
                    0,
                    0.008,
                    0,
                    0.039,
                    0.029,
                ]
            }
        )
        pd.testing.assert_frame_equal(actual_df, expected_df)

    def test_calculate_return_on_employed_capital_two_toy_columns(self) -> None:
        one_way_transaction_cost_in_basis_points = 10
        input_df = pd.DataFrame(
            {"A": [0.0, 0.01, 0, 0.04, 0.03], "B": [0, -0.35, -0.19, 0, 0.2]}
        )
        actual_df = calculate_daily_return_on_employed_capital(
            input_df, one_way_transaction_cost_in_basis_points
        )
        expected_df = pd.DataFrame(
            {
                PORTFOLIO_DAILY_RETURN_AFTER_T_COST_COLUMN_NAME: [
                    0,
                    -0.173,
                    -0.191,
                    0.039,
                    0.112,
                ]
            }
        )
        pd.testing.assert_frame_equal(actual_df, expected_df)
    

