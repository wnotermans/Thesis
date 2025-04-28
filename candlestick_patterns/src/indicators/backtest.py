import os
import time

import pandas as pd

from shared import constants

COLUMN_HEADERS = [
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
    "%R",
]


def backtest_indicators(*, run_name: str) -> None:
    t = time.perf_counter()
    print("Backtesting indicators", end="\r")
    dataframe_rows = []
    for number_str in constants.PATTERN_NUMBERS_AS_STRING:
        base_path = f"data/runs/{run_name}"
        for pattern in os.listdir(f"{base_path}/evaluation/{number_str}"):
            if not pattern.endswith("indicators.csv"):
                continue
            dataframe_rows.append(
                pd.read_csv(
                    f"{base_path}/evaluation/{number_str}/{pattern}",
                    header=None,
                )
                .iloc[0]
                .to_list()
            )
    pd.DataFrame(dataframe_rows, columns=COLUMN_HEADERS).describe().round(2).to_csv(
        f"{base_path}/backtest.csv"
    )
    print(f"Backtesting indicators done in {time.perf_counter() - t:.2f}s", end="\n\n")
