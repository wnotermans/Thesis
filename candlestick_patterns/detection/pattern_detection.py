import os
import time

import numpy as np
import pandas as pd
import pyarrow as pa
from word2number import w2n

from detection.patterns import (
    eight_patterns,
    eleven_patterns,
    five_patterns,
    four_patterns,
    one_patterns,
    ten_patterns,
    thirteen_patterns,
    three_patterns,
    twelve_patterns,
    two_patterns,
)
from detection.patterns.functions import candlestick_functions as cf


def detection(df: pd.DataFrame, percentile: tuple, mode: str = "exclude") -> None:
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
        f"{col}_{shift}": df[col].shift(shift).values
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

    num_patterns = 309
    i = 0

    for number in [
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
    ]:

        pattern_funcs = []  # get all functions names of the pattern functions
        for name in dir(globals().get(f"{number}_patterns")):
            attr = getattr(globals().get(f"{number}_patterns"), name)
            if callable(attr):
                pattern_funcs.append(name)

        for func_name in pattern_funcs:

            try:
                os.remove(f"data/patterns/{number}/{func_name}.parquet")
            except FileNotFoundError:
                pass

            print(
                f"Detecting {func_name:<53} <|> "
                + f"{'-'*(48*i//num_patterns)+">":<49} "
                + f"({i+1:>3}/{num_patterns})",
                end="\r",
            )
            i += 1

            func_call = getattr(globals().get(f"{number}_patterns"), func_name)
            pat = func_call(locals().get(f"{number}_candle"), T, percentile)
            pat = handle_gaps(pat, df["gap"], w2n.word_to_num(number), mode=mode)

            pa.parquet.write_table(
                pa.table({f"{func_name}": pat}),
                f"data/patterns/{number}/{func_name}.parquet",
                compression="LZ4",
            )

    print()
    print(
        f"All done. Total detection time: {round(time.perf_counter()-total_time,2)}s",
        end="\n\n",
    )


def handle_gaps(
    pattern: pd.Series, gap: pd.Series, number_candles: int, mode: str = "exclude"
) -> pd.Series:
    """
    Handle gaps in the data according to the given mode.

    * "exclude": excludes patterns where there are gaps inbetween the candles that make
    up the pattern.
    * "ignore": ignore any gaps, replace any NaNs in the data with False.
    * "only": only consideres patterns with gaps inbetween the candles.

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
