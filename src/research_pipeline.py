from datetime import datetime

from src.pair_selection import (
    get_cointegrated_pairs_with_hurst_exponent
)

ONE_YEAR_IN_TRADING_DAYS = 252

cointegration_test_period_in_trading_days = ONE_YEAR_IN_TRADING_DAYS
beta_estimation_period_in_trading_days = ONE_YEAR_IN_TRADING_DAYS * 2
end_date_for_cointegration_test_period = datetime(2023, 12, 31)

cointegrated_pairs_with_hurst_exponent = get_cointegrated_pairs_with_hurst_exponent(
    end_date_for_cointegration_test_period,
    beta_estimation_period_in_trading_days,
    cointegration_test_period_in_trading_days
)






