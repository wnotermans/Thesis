import time

import numpy as np
import pandas as pd
import pyarrow as pa

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


def run(df):

    total_time = time.time()

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
        print(f"Candlestick patterns with {number} candlestick(s)")

        pattern_funcs = []  # get all functions names of the pattern functions
        for name in dir(globals().get(f"{number}_patterns")):
            attr = getattr(globals().get(f"{number}_patterns"), name)
            if callable(attr):
                pattern_funcs.append(name)

        i = 1  # counter for total number of patterns
        num_funcs = len(pattern_funcs)

        t = time.time()
        for func_name in pattern_funcs:
            print(
                f"Detecting {func_name:<54} | "
                + f"{'#'*(50*i//num_funcs):<50} "
                + f"({i:>3}/{num_funcs})",
                end="\r",
            )

            func_call = getattr(globals().get(f"{number}_patterns"), func_name)
            pat = func_call(locals().get(f"{number}_candle"), T)

            pa.parquet.write_table(
                pa.table({f"{func_name}": pat}),
                f"../data/patterns/{number}/{func_name}.parquet",
                compression="LZ4",
            )

            i += 1

        print()
        print(
            f"Detecting patterns with {number} candlestick(s): "
            + f"Done in {round(time.time()-t,2):<3.2f}s.",
            end="\n\n",
        )

    print(
        f"All done. Total detection time: {round(time.time()-total_time,2)}s",
        end="\n\n",
    )
