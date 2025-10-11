from src.backtesting import (
    calculate_number_of_trades_per_day
    )
import pandas as pd
import numpy as np

class TestCalculateNumberOfTradesPerDay:

    def test_calculate_num_trades_per_day_toy_data(self) -> None:
        input_df = pd.DataFrame(
            {
                'A' : [0, 0, 1, 1, 1, 0, 0, 1, 1],
                'B' : [0, 1, 0, 1, 0, 1, 0, 0, 1],
                'C' : [0, 1, 1, 1, 0, 1, 1, 1, 0],
            }
        )
        actual = calculate_number_of_trades_per_day(input_df)
        expected = pd.Series(
            [0, 3, 1, 3, 1, 3, 0, 2, 3]
        )
        pd.testing.assert_series_equal(actual.astype('int64'), expected)