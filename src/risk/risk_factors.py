from src.risk.risk_factor import RiskFactor

MARKET_RISK_US = RiskFactor("fama_french_market_risk_us")
HIGH_MINUS_LOW_US = RiskFactor("fama_french_high_minus_low_us")
SMALL_MINUS_BIG_US = RiskFactor("fama_french_small_minus_big_us")
CONSERVATIVE_MINUS_AGGRESSIVE_US = RiskFactor("fama_french_conservative_minus_aggressive_us")
ROBUS_MINUS_WEAK_US = RiskFactor("fama_french_robust_minus_weak_us")