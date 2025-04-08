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
ADDITIONAL_FILTERS_DICT = {}
TREND_AVERAGING_METHOD = "MA"  # MA, WMA, EWMA
TREND_AVERAGING_METHOD_KWARGS = {}  # span=5, consecutive=7
TREND_DECISION_METHOD = "monotonic"  # monotonic, counting, high_low
TREND_DECISION_METHOD_KWARGS = {}  # counting: fraction=0.7
DATA_GAP_HANDLING = "exclude"
EVALUATION_METHOD = "stop_loss_take_profit"
SHARED_PARAMS_DICT = {
    "start_end_time": START_END_TIME,
    "filter_news": FILTER_NEWS,
    "filter_news_kwargs": FILTER_NEWS_KWARGS,
    "additional_filters_dict": ADDITIONAL_FILTERS_DICT,
    "trend_averaging_method": TREND_AVERAGING_METHOD,
    "trend_averaging_method_kwargs": TREND_AVERAGING_METHOD_KWARGS,
    "trend_decision_method": TREND_DECISION_METHOD,
    "trend_decision_method_kwargs": TREND_DECISION_METHOD_KWARGS,
    "data_gap_handling": DATA_GAP_HANDLING,
    "evaluation_method": EVALUATION_METHOD,
}
