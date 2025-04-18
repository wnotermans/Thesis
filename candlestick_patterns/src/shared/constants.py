# Pretty self-explanatory.
TOTAL_NUMBER_OF_PATTERNS = 315
THREE_STAR_SIGNIFICANCE = 0.001
TWO_STAR_SIGNIFICANCE = 0.01
ONE_STAR_SIGNIFICANCE = 0.05
PATTERN_NUMBERS_AS_STRING = [
    "one",
    "two",
    "three",
    "four",
    "five",
    "eight",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
]

# Percentiles to use for the splitting of body/shadow sizes into bins.
# These are doji, small, normal and tall for body;
# and none, small, normal, long and extremely long for shadows.
# Source: The classification of candlestick charts: laying the foundation for further
# empirical research, https://doi.org/10.1007/3-540-31314-1_64
BODY_PERCENTILES = [10, 30, 70]
SHADOW_PERCENTILES = [10, 30, 70, 90]

# Margins of the stop_loss_take_profit evaluation.
STOP_LOSS_MARGIN_PERCENT = 1

# Amount of years of data necessary for a dataset to be considered "of proper length".
MINIMAL_FULL_DATA_YEARS = 15

# Patterns with less than this amount of detections are never considered significant.
MINIMAL_SIGNIFICANT_DETECTION_SIZE = 100

# Run settings
FILENAMES = ["Wiener_small"]
INTERVAL_MINUTES = [10]
START_END_TIME = ("09:30:00", "16:00:00")
FILTER_NEWS = False
FILTER_NEWS_KWARGS = {}
INDICATOR_KWARGS = {}
INDICATOR_DEFAULTS = {  # Avoids bloating the parameters file
    "ADX": {"window": 15},
    "ATR": {"window": 15},
    "BB": {"window": 20},
    "DPO": {"window": 20},
    "MA": {"window": 7},
    "MACD": {"spans": (5, 11, 5)},
    "MFI": {"window": 15},
    "momentum": {"span": 7},
    "PSAR": {"step": 0.02, "max_accel_factor": 0.2},
    "RSI": {"window": 15},
    "TRIX": {"windows": (15, 15, 15)},
    "VI": {"window": 21},
    "%R": {"window": 14},
}
TREND_AVERAGING_METHOD = "SMA"  # SMA, WMA, EMA
TREND_AVERAGING_METHOD_KWARGS = {}  # span=5, consecutive=7
TREND_AVERAGING_DEFAULTS = {
    "SMA": {"window": 5},
    "WMA": {"window": 5},
    "EMA": {"window": 5},
}
TREND_DECISION_METHOD = "monotonic"  # monotonic, counting, high_low, PSAR
TREND_DECISION_METHOD_KWARGS = {}  # counting: fraction=0.7
TREND_DECISION_DEFAULTS = {
    "monotonic": {"span": 7},
    "counting": {"span": 7, "fraction": 0.7},
    "PSAR": {"step": 0.02, "max_accel_factor": 0.2},
}
DATA_GAP_HANDLING = "exclude"
EVALUATION_METHOD = "stop_loss_take_profit"
SHARED_PARAMS_DICT = {
    "start_end_time": START_END_TIME,
    "filter_news": FILTER_NEWS,
    "filter_news_kwargs": FILTER_NEWS_KWARGS,
    "additional_filters_dict": INDICATOR_KWARGS,
    "trend_averaging_method": TREND_AVERAGING_METHOD,
    "trend_averaging_method_kwargs": TREND_AVERAGING_METHOD_KWARGS,
    "trend_decision_method": TREND_DECISION_METHOD,
    "trend_decision_method_kwargs": TREND_DECISION_METHOD_KWARGS,
    "data_gap_handling": DATA_GAP_HANDLING,
    "evaluation_method": EVALUATION_METHOD,
}
