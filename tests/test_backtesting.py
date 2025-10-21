import pandas as pd
from src.backtesting import (
    calculate_number_of_trades_per_day,
    calculate_daily_return_on_employed_capital_after_t_costs,
    calculate_daily_portfolio_return_before_t_costs,
    create_trade_signals_from_z_scored_spread,
    DAILY_TRANSACTION_COUNT_COLUMN_NAME,
    PORTFOLIO_DAILY_RETURN_AFTER_T_COST_COLUMN_NAME,
)


class TestCreateTradeSignalsFromSpreadRollingZScore:

    def test_create_trade_signals_from_z_scored_spread(self) -> None:
        ticker_one = "KVUE"
        ticker_two = "WVE"
        z_scored_spread_series = pd.read_csv(
            "tests/test_data/KVUE_WVE_z_scored_spread.csv",
            index_col=0,
            parse_dates=True,
        ).squeeze("columns")

        actual_df = create_trade_signals_from_z_scored_spread(
            ticker_one,
            ticker_two,
            z_scored_spread_series,
            exit_threshold_proximity_tolerance=0.1,
        )
        expected_df = pd.read_csv(
            "tests/test_data/KVUE_WVE_trade_signals.csv", index_col=0, parse_dates=True
        )
        pd.testing.assert_frame_equal(actual_df, expected_df)


class TestCalculateNumberOfTradesPerDay:

    def test_calculate_num_trades_per_day_toy_data(self) -> None:
        toy_input_df = pd.DataFrame(
            {
                "A": [0, 0, 1, 1, 1, 0, 0, 1, 1],
                "B": [0, 1, 0, 1, 0, 1, 0, 0, 1],
                "C": [0, 1, 1, 1, 0, 1, 1, 1, 0],
            }
        )
        actual_df = calculate_number_of_trades_per_day(toy_input_df)
        expected_df = pd.DataFrame(
            {DAILY_TRANSACTION_COUNT_COLUMN_NAME: [0, 3, 1, 3, 1, 3, 0, 2, 3]}
        )
        pd.testing.assert_frame_equal(actual_df, expected_df)

    def test_calculate_num_trades_per_day_sample_data(self) -> None:
        trade_returns_for_tickers_df = pd.read_csv(
            "tests/test_data/trade_returns_for_tickers.csv",
            index_col=0,
            parse_dates=True,
        )
        actual_df = calculate_number_of_trades_per_day(trade_returns_for_tickers_df)
        expected_df = pd.read_csv(
            "tests/test_data/num_trades_per_day_expected.csv",
            index_col=0,
            parse_dates=True,
        )
        pd.testing.assert_frame_equal(actual_df, expected_df)


class TestCalculateReturnOnEmployedCaptial:

    def test_calculate_daily_portfolio_return_before_t_costs(self) -> None:
        expected_df = pd.read_csv(
            "tests/test_data/portfolio_daily_return_before_t_cost.csv",
            index_col=0,
            parse_dates=True,
        )
        trade_returns_for_tickers_df = pd.read_csv(
            "tests/test_data/trade_returns_for_tickers_2.csv",
            index_col=0,
            parse_dates=True,
        )
        actual_df = calculate_daily_portfolio_return_before_t_costs(
            trade_returns_for_tickers_df
        )
        pd.testing.assert_frame_equal(expected_df, actual_df)

    def test_calculate_return_on_employed_capital_after_t_cost_one_toy_column(
        self,
    ) -> None:
        one_way_transaction_cost_in_basis_points = 10
        input_df = pd.DataFrame({"A": [0.0, 0.01, 0, 0.04, 0.03]})
        actual_df = calculate_daily_return_on_employed_capital_after_t_costs(
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

    def test_calculate_return_on_employed_capital_after_t_cost_two_toy_columns(
        self,
    ) -> None:
        one_way_transaction_cost_in_basis_points = 10
        input_df = pd.DataFrame(
            {"A": [0.0, 0.01, 0, 0.04, 0.03], "B": [0, -0.35, -0.19, 0, 0.2]}
        )
        actual_df = calculate_daily_return_on_employed_capital_after_t_costs(
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

    def test_calculate_return_on_employed_capital_after_t_costs_sample_data(
        self,
    ) -> None:
        one_way_t_cost_in_basis_points = 10
        expected_df = pd.read_csv(
            "tests/test_data/daily_return_on_employed_capital_after_t_cost_sample.csv",
            index_col=0,
            parse_dates=True,
        )
        trade_returns_for_all_tickers_df = pd.read_csv(
            "tests/test_data/trade_returns_for_tickers_2.csv",
            index_col=0,
            parse_dates=True,
        )
        actual_df = calculate_daily_return_on_employed_capital_after_t_costs(
            trade_returns_for_all_tickers_df,
            one_way_t_cost_in_basis_points=one_way_t_cost_in_basis_points,
        )
        pd.testing.assert_frame_equal(actual_df, expected_df)
