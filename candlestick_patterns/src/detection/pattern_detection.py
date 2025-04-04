import time

import numpy as np
import pandas as pd
import pyarrow.parquet
from word2number import w2n

from detection.patterns import (
    eight_patterns,  # noqa: F401
    eleven_patterns,  # noqa: F401
    five_patterns,  # noqa: F401
    four_patterns,  # noqa: F401
    one_patterns,  # noqa: F401
    ten_patterns,  # noqa: F401
    thirteen_patterns,  # noqa: F401
    three_patterns,  # noqa: F401
    twelve_patterns,  # noqa: F401
    two_patterns,  # noqa: F401
)
from shared import constants, shared_functions


def detection(
    df: pd.DataFrame, percentile: tuple, data_gap_handling: str, *, run_name: str
) -> None:
    """
    Performs pattern detection.

    Parameters
    ----------
    df : pd.DataFrame
        A Dataframe with OHLC data.
    percentile : tuple
        Tuple of length percentiles.
    mode : {"exclude", "ignore", "only"}
        Mode of handling gaps in the data.

    Returns
    -------
    None
        Outputs the 309 candlestick patterns to disk in `.parquet` format.
    """
    total_time = time.perf_counter()

    column_dict = {
        f"{col}_{shift}": df[col].shift(shift).to_numpy()
        for col in ["open", "high", "low", "close", "volume"]
        for shift in range(13)
    }

    candle_dict = {
        f"candle_minus_{n}": np.array(
            [
                column_dict[f"open_{n}"],
                column_dict[f"high_{n}"],
                column_dict[f"low_{n}"],
                column_dict[f"close_{n}"],
                column_dict[f"volume_{n}"],
            ]
        ).T
        for n in range(13)
    }

    T = np.array(df["trend"].values)

    one_candle = candle_dict["candle_minus_0"]
    two_candle = [candle_dict[f"candle_minus_{n}"] for n in range(1, -1, -1)]
    three_candle = [candle_dict[f"candle_minus_{n}"] for n in range(2, -1, -1)]
    four_candle = [candle_dict[f"candle_minus_{n}"] for n in range(3, -1, -1)]
    five_candle = [candle_dict[f"candle_minus_{n}"] for n in range(4, -1, -1)]
    eight_candle = [candle_dict[f"candle_minus_{n}"] for n in range(7, -1, -1)]
    ten_candle = [candle_dict[f"candle_minus_{n}"] for n in range(9, -1, -1)]
    eleven_candle = [candle_dict[f"candle_minus_{n}"] for n in range(10, -1, -1)]
    twelve_candle = [candle_dict[f"candle_minus_{n}"] for n in range(11, -1, -1)]
    thirteen_candle = [candle_dict[f"candle_minus_{n}"] for n in range(12, -1, -1)]

    i = 0

    for number in constants.PATTERN_NUMBERS_AS_STRING:
        func_name_list = extract_func_names(number_candles=number)

        for func_name in func_name_list:
            shared_functions.print_status_bar(
                func_name, i, constants.TOTAL_NUMBER_OF_PATTERNS
            )

            i += 1

            func_call = getattr(globals().get(f"{number}_patterns"), func_name)
            patterns = func_call(locals().get(f"{number}_candle"), T, percentile)
            patterns_gaps_handled = handle_gaps(
                patterns, df["gap"], w2n.word_to_num(number), mode=data_gap_handling
            )
            table = pyarrow.Table.from_arrays(
                [patterns_gaps_handled], names=["pattern"]
            )
            pyarrow.parquet.write_table(
                table,
                f"data/runs/{run_name}/detection/{number}/{func_name}.parquet",
                compression="brotli",
            )

    print()
    print(
        f"All done. Total detection time: {
            round(time.perf_counter() - total_time, 2)
        }s",
        end="\n\n",
    )


def extract_func_names(number_candles: str) -> list[str]:
    """
    Extract the function names of pattern functions from number of candlesticks.

    Parameters
    ----------
    number_candles : str
        The number of candlesticks in the pattern.

    Returns
    -------
    list[str]
        List of all the function names of the candlestick patterns.
    """
    func_names = []
    file_name = f"{number_candles}_patterns"
    for name in dir(globals().get(file_name)):
        attribute = getattr(globals().get(file_name), name)
        if callable(attribute):
            func_names.append(name)
    return func_names


def handle_gaps(
    pattern: pd.Series, gap: pd.Series, number_candles: int, mode: str
) -> np.ndarray:
    """
    Handle gaps in the data according to the given mode.

    * "exclude": excludes patterns where there are gaps between the candles that make
    up the pattern.
    * "ignore": ignore any gaps, replace any NaNs in the data with False.
    * "only": only considers patterns with gaps between the candles.

    Parameters
    ----------
    pattern : np.ndarray
        Boolean series with the candlestick pattern.
    gap : pd.Series
        Boolean series with the data gaps.
    number_candles : int
        Number of candles in the pattern
    mode : {"exclude", "ignore", "only"}
        Gap handling mode.

    Returns
    -------
    np.ndarray
        Boolean series with patterns included or excluded according to the gap handling
        policy.
    """
    if mode == "ignore":
        return pattern
    gap_number_candles_adjusted = np.logical_or.reduce(
        [np.array(gap.shift(n)) for n in range(number_candles)]
    )
    if mode == "exclude":
        return np.logical_and(pattern, np.logical_not(gap_number_candles_adjusted))
    if mode == "only":
        return np.logical_and(pattern, gap_number_candles_adjusted)
    return None
