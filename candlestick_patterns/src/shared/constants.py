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
INDICATOR_COLUMNS = [
    "ATR",
    "ADX",
    "BB_low",
    "BB_mid",
    "BB_high",
    "DPO",
    "MA",
    "MACD",
    "MACD_signal",
    "MFI",
    "momentum",
    "PSAR",
    "RSI",
    "TRIX",
    "VI+",
    "VI-",
    "VI_diff",
    "VW",
    "%R",
]

# Percentiles to use for the splitting of body/shadow sizes into bins.
# These are doji, small, normal and tall for body;
# and none, small, normal, long and extremely long for shadows.
# Source: The classification of candlestick charts: laying the foundation for further
# empirical research, https://doi.org/10.1007/3-540-31314-1_64
BODY_PERCENTILES = [10, 30, 70]
SHADOW_PERCENTILES = [10, 30, 70, 90]


# Amount of years of data necessary for a dataset to be considered "of proper length".
MINIMAL_FULL_DATA_YEARS = 15

# Patterns with less than this amount of detections are never considered significant.
MINIMAL_SIGNIFICANT_DETECTION_SIZE = 100

########################################################################################
###### Run settings, defaults are used to avoid bloating the parameters.txt file #######
########################################################################################

# Data set(s) (loops if multiple)
FILENAMES = ["Wiener_small"]
# Time intervals (loops if multiple)
INTERVAL_MINUTES = [5]
# Filter down to this time range
START_END_TIME = ("9:30:00", "16:00:00")
# Filters by economic news
FILTER_NEWS_KWARGS = {}  # "impact_level": ("MEDIUM", "HIGH")
FILTER_NEWS_DEFAULTS = {"minutes_after": 60}
# Override
INDICATOR_KWARGS = {}
# Defaults
INDICATOR_DEFAULTS = {
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
    "VW": {"minutes": 30},
    "%R": {"window": 14},
}
# Filter by indicators
INDICATOR_FILTER_KWARGS = {}
# Which averaging method to use in trend calculation; options are: SMA, WMA, EMA
TREND_AVERAGING_METHOD = "SMA"
# Override
TREND_AVERAGING_METHOD_KWARGS = {}
# Defaults
TREND_AVERAGING_DEFAULTS = {
    "SMA": {"window": 5},
    "WMA": {"window": 5},
    "EMA": {"window": 5},
}
# Which trend decision method to use; options are: monotonic, counting, high_low, PSAR
TREND_DECISION_METHOD = "monotonic"
# Override
TREND_DECISION_METHOD_KWARGS = {}
# Defaults
TREND_DECISION_DEFAULTS = {
    "monotonic": {"span": 7},
    "counting": {"span": 7, "fraction": 0.7},
    "PSAR": {"step": 0.02, "max_accel_factor": 0.2},
}
# How to handle gaps in the data
DATA_GAP_HANDLING = "exclude"
# Margins of the stop_loss_take_profit evaluation.
STOP_LOSS_TAKE_PROFIT_MARGINS = {"ATR": None}

# Dict that is included in parameters.txt
SHARED_PARAMS_DICT = {
    "start_end_time": START_END_TIME,
    "filter_news_kwargs": FILTER_NEWS_KWARGS,
    "indicator_kwargs": INDICATOR_KWARGS,
    "indicator_filter_kwargs": INDICATOR_FILTER_KWARGS,
    "trend_averaging_method": TREND_AVERAGING_METHOD,
    "trend_averaging_method_kwargs": TREND_AVERAGING_METHOD_KWARGS,
    "trend_decision_method": TREND_DECISION_METHOD,
    "trend_decision_method_kwargs": TREND_DECISION_METHOD_KWARGS,
    "data_gap_handling": DATA_GAP_HANDLING,
    "stop_loss_take_profit_margins": STOP_LOSS_TAKE_PROFIT_MARGINS,
}
