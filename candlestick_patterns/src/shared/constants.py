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
