import csv
import os

import numba
import numpy as np
import pandas as pd
import pyarrow.parquet

from shared import constants, shared_functions


def stop_loss_take_profit_evaluation(
    df: pd.DataFrame, margins: dict[str, float | None], *, run_name: str, split: int
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
    open_array = df["open"].to_numpy()
    high_array = df["high"].to_numpy()
    low_array = df["low"].to_numpy()
    ATR_array = df["ATR"].to_numpy()

    i = 0
    for number_str in constants.PATTERN_NUMBERS_AS_STRING:
        for pattern in os.listdir(f"data/runs/{run_name}/detection/{number_str}"):
            shared_functions.print_status_bar(
                pattern.removesuffix(".parquet"),
                i,
                constants.TOTAL_NUMBER_OF_PATTERNS,
                split,
            )

            i += 1

            df["pattern"] = (
                pyarrow.parquet.read_table(
                    f"data/runs/{run_name}/detection/{number_str}/{pattern}"
                )
                .to_pandas()
                .set_index(df.index)
                .shift(1)
            ).astype(bool)
            df.loc[df.index[0], "pattern"] = False
            num_detected = df["pattern"].sum()

            csv_path = (
                f"data/runs/{run_name}/evaluation/{number_str}/"
                f"{pattern.removesuffix('.parquet')}"
            )

            if split == 1:
                with open(f"{csv_path}_evaluation.csv", "w") as _:
                    pass

            if num_detected <= constants.MINIMAL_SIGNIFICANT_DETECTION_SIZE:
                csv_data = {
                    f"{csv_path}_evaluation.csv": ["/"] * 2 + [0],
                }
                write_csvs(csv_data)

            else:
                if all(
                    method not in margins
                    for method in ["ATR", "constant", "percentage"]
                ):
                    raise ValueError(
                        "No correct margin method specified, choose either "
                        "'ATR', 'constant' or 'percentage'"
                    )
                bool_array = df["pattern"].to_numpy()
                non_pattern = ~bool_array
                if non_pattern.sum() > (N := len(non_pattern) // 3):
                    selected_indices = np.random.choice(
                        np.where(non_pattern)[0], size=N, replace=False
                    )
                    null_array = np.zeros_like(bool_array, dtype=bool)
                    null_array[selected_indices] = True
                else:
                    null_array = non_pattern
                if "percentage" in margins:
                    eval_list = find_first_breakthroughs_percent(
                        (bool_array, open_array, high_array, low_array),
                        margins["percentage"],
                    )
                    null_list = find_first_breakthroughs_percent(
                        (null_array, open_array, high_array, low_array),
                        margins["percentage"],
                    )
                elif "constant" in margins:
                    eval_list = find_first_breakthroughs_constant(
                        (bool_array, open_array, high_array, low_array),
                        margins["constant"],
                    )
                    null_list = find_first_breakthroughs_constant(
                        (null_array, open_array, high_array, low_array),
                        margins["constant"],
                    )
                else:
                    eval_list = find_first_breakthroughs_ATR(
                        (bool_array, open_array, high_array, low_array, ATR_array)
                    )
                    null_list = find_first_breakthroughs_ATR(
                        (null_array, open_array, high_array, low_array, ATR_array)
                    )

                df["evaluation"] = None
                df.loc[df["pattern"], "evaluation"] = eval_list
                del df["pattern"]

                wins = int(np.nansum(eval_list))
                number_detected = len(eval_list)
                win_rate = wins / number_detected
                absolute_win_rate = (
                    str(win_rate) + "+"
                    if win_rate >= 0.5  # noqa: PLR2004
                    else str(0.5 + abs(0.5 - win_rate)) + "-"
                )

                null_win = int(np.nansum(null_list)) / len(null_list)
                null_win = (
                    str(null_win) + "+"
                    if null_win >= 0.5  # noqa: PLR2004
                    else str(0.5 + abs(0.5 - null_win)) + "-"
                )

                csv_data = {
                    f"{csv_path}_evaluation.csv": [
                        absolute_win_rate,
                        null_win,
                        number_detected,
                    ],
                }
                write_csvs(csv_data)


def write_csvs(csv_data: dict) -> None:
    """
    Write csv data to disk.

    Parameters
    ----------
    csv_data : dict
        Dict containing filepaths as keys and data as values.
    """
    for path, data in csv_data.items():
        with open(path, "a") as csvfile:
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
