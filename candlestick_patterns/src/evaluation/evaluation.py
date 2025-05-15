import csv
import os
import time

import numba
import numpy as np
import pandas as pd
import pyarrow.parquet
from scipy.stats import binomtest

from shared import constants, shared_functions


def stop_loss_take_profit_evaluation(
    df: pd.DataFrame, margins: dict[str, float | None], *, run_name: str
) -> None:
    """
    Stop loss/take profit-based candlestick pattern evaluation.

    For every time a pattern is detected, checks which margin is triggered first:
    the stop loss or the take profit one.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame with OHLC data. Candlestick patterns are read from disk.

    Returns
    -------
    None
        Win %, "less" and "greater" binomial tests to disk.
    """
    t = time.perf_counter()
    open_array = df["open"].to_numpy()
    high_array = df["high"].to_numpy()
    low_array = df["low"].to_numpy()
    ATR_array = df["ATR"].to_numpy()

    i = 0
    for number_str in constants.PATTERN_NUMBERS_AS_STRING:
        for pattern in os.listdir(f"data/runs/{run_name}/detection/{number_str}"):
            shared_functions.print_status_bar(
                pattern.removesuffix(".parquet"), i, constants.TOTAL_NUMBER_OF_PATTERNS
            )

            i += 1

            df["pattern"] = (
                pyarrow.parquet.read_table(
                    f"data/runs/{run_name}/detection/{number_str}/{pattern}"
                )
                .to_pandas()
                .set_index(df.index)
                .shift(1)
            )
            df.loc[df.index[0], "pattern"] = False
            num_detected = df["pattern"].sum()

            csv_path = (
                f"data/runs/{run_name}/evaluation/{number_str}/"
                f"{pattern.removesuffix('.parquet')}"
            )

            if num_detected <= constants.MINIMAL_SIGNIFICANT_DETECTION_SIZE:
                csv_data = {
                    f"{csv_path}evaluation.csv": ["/"] * 3 + [0],
                    f"{csv_path}indicators.csv": [""]
                    * len(constants.INDICATOR_COLUMNS),
                }
                write_csvs(csv_data)

            else:
                bool_array = df["pattern"].astype(bool).to_numpy()
                if all(
                    method not in margins
                    for method in ["ATR", "constant", "percentage"]
                ):
                    raise ValueError(
                        "No correct margin method specified, choose either "
                        "'ATR', 'constant' or 'percentage'"
                    )
                if "percentage" in margins:
                    eval_list = find_first_breakthroughs_percent(
                        (bool_array, open_array, high_array, low_array),
                        margins["percentage"],
                    )
                elif "constant" in margins:
                    eval_list = find_first_breakthroughs_constant(
                        (bool_array, open_array, high_array, low_array),
                        margins["constant"],
                    )
                else:
                    eval_list = find_first_breakthroughs_ATR(
                        (bool_array, open_array, high_array, low_array, ATR_array)
                    )

                df["evaluation"] = None
                df.loc[df["pattern"], "evaluation"] = eval_list
                del df["pattern"]

                success_indicator_means = df.loc[
                    df["evaluation"] == 1, constants.INDICATOR_COLUMNS
                ].mean()

                wins = int(np.nansum(eval_list))
                number_detected = len(eval_list)
                win_rate = wins / number_detected
                absolute_win_rate = (
                    win_rate
                    if win_rate >= 0.5  # noqa: PLR2004
                    else 0.5 + abs(0.5 - win_rate)
                )

                down_test = binomtest(
                    wins,
                    number_detected,
                    p=0.5,
                    alternative="less",
                ).pvalue
                up_test = binomtest(
                    wins,
                    number_detected,
                    p=0.5,
                    alternative="greater",
                ).pvalue

                csv_data = {
                    f"{csv_path}evaluation.csv": [
                        absolute_win_rate,
                        down_test,
                        up_test,
                        number_detected,
                    ],
                    f"{csv_path}indicators.csv": success_indicator_means,
                }
                write_csvs(csv_data)
    print()
    print(
        f"All done. Total evaluation time: {time.perf_counter() - t:3.2f}s",
        end="\n\n",
    )


def write_csvs(csv_data: dict) -> None:
    """
    Write csv data to disk.

    Parameters
    ----------
    csv_data : dict
        Dict containing filepaths as keys and data as values.
    """
    for path, data in csv_data.items():
        with open(path, "w") as csvfile:
            csv.writer(csvfile).writerow(data)


@numba.jit
def find_first_breakthroughs_percent(
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    percentage: float,
) -> list:
    """
    Find the stop loss take profit breakthroughs. Margins are percentage-based.

    Parameters
    ----------
    arrays : tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Boolean array of the patterns; open, high and low array.
    percentage : float
        Percentage that defines the margins.

    Returns
    -------
    list
        Evaluation: 0 for bottom breakthrough, 1 for upper, ``np.nan`` else.
    """

    bool_array, open_array, high_array, low_array = arrays

    breaches = []
    true_indices = np.flatnonzero(bool_array)

    for i in true_indices:
        start_value = open_array[i]
        upper_threshold = start_value * (1 + percentage / 100)
        lower_threshold = start_value * (1 - percentage / 100)

        for j in range(i + 1, len(open_array)):
            if high_array[j] > upper_threshold:
                breaches.append(1)
                break
            if low_array[j] < lower_threshold:
                breaches.append(0)
                break
        else:
            breaches.append(np.nan)

    return breaches


@numba.jit
def find_first_breakthroughs_constant(
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    constant: float,
) -> list:
    """
    Find the stop loss take profit breakthroughs. Margins are constant.

    Parameters
    ----------
    arrays : tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Boolean array of the patterns; open, high and low array.
    constant : float
        Percentage that defines the margins.

    Returns
    -------
    list
        Evaluation: 0 for bottom breakthrough, 1 for upper, ``np.nan`` else.
    """

    bool_array, open_array, high_array, low_array = arrays

    breaches = []
    true_indices = np.flatnonzero(bool_array)

    for i in true_indices:
        start_value = open_array[i]
        upper_threshold = start_value + constant
        lower_threshold = start_value - constant

        for j in range(i + 1, len(open_array)):
            if high_array[j] > upper_threshold:
                breaches.append(1)
                break
            if low_array[j] < lower_threshold:
                breaches.append(0)
                break
        else:
            breaches.append(np.nan)

    return breaches


@numba.jit
def find_first_breakthroughs_ATR(
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> list:
    """
    Find the stop loss take profit breakthroughs. Margins are percentage-based and
    controlled globally.

    Parameters
    ----------
    arrays : tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Boolean array of the patterns; open, high and low array; ATR array.

    Returns
    -------
    list
        Evaluation: 0 for bottom breakthrough, 1 for upper, ``np.nan`` else.
    """

    bool_array, open_array, high_array, low_array, ATR_array = arrays

    breaches = []
    true_indices = np.flatnonzero(bool_array)

    for i in true_indices:
        start_value = open_array[i]
        upper_threshold = start_value + ATR_array[i]
        lower_threshold = start_value - ATR_array[i]

        for j in range(i + 1, len(open_array)):
            if high_array[j] > upper_threshold:
                breaches.append(1)
                break
            if low_array[j] < lower_threshold:
                breaches.append(0)
                break
        else:
            breaches.append(np.nan)

    return breaches
