import csv
import os
import time

import numba
import numpy as np
import pandas as pd
import pyarrow.parquet
from scipy.stats import binomtest


def evaluation(
    df: pd.DataFrame, detection_method: str = "stop_loss_take_profit", *, run_name: str
) -> None:
    """
    Performs evaluation of the patterns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to be analyzed
    detection_method : str, optional, default "stop_loss_take_profit"
        Which detection method to use.
    """
    if detection_method == "stop_loss_take_profit":
        stop_loss_take_profit_evaluation(df, run_name=run_name)
    if detection_method == "n_holding_periods":
        n_holding_periods(df, run_name=run_name)


def stop_loss_take_profit_evaluation(df: pd.DataFrame, *, run_name: str) -> None:
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

    HL_ARRAY = df[["high", "low"]].to_numpy()
    MARGIN_PERCENT = 1

    total_time = time.perf_counter()
    NUM_PATTERNS = 315
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
        for pattern in os.listdir(f"data/runs/{run_name}/detection/{number}"):
            print_status_bar(pattern, i, NUM_PATTERNS)

            i += 1

            df["pat"] = (
                pyarrow.parquet.read_table(
                    f"data/runs/{run_name}/detection/{number}/{pattern}"
                )
                .to_pandas()
                .set_index(df.index)
                .shift(1)
            )
            df.loc[df.index[0], "pat"] = False
            num_detected = df["pat"].sum()

            if num_detected in (0, 1):
                with open(
                    f"data/runs/{run_name}/evaluation/{number}/"
                    f"{pattern.removesuffix('.parquet')}.csv",
                    "w",
                ) as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["/"] * 3)

            else:
                pattern_indices = df[df["pat"]].index
                del df["pat"]

                eval_list = np.array([])

                for pattern_index in pattern_indices:
                    open_index = df.index.get_loc(pattern_index)
                    OP = df.loc[pattern_index, "open"]

                    eval_list = np.append(
                        eval_list,
                        find_first_breakthrough(
                            HL_ARRAY, OP, open_index, len(df), MARGIN_PERCENT
                        ),
                    )

                eval_str = f"{np.nansum(eval_list) / len(eval_list):.2%}"
                down_test = binomtest(
                    int(np.nansum(eval_list)),
                    len(eval_list),
                    p=0.5,
                    alternative="less",
                ).pvalue
                up_test = binomtest(
                    int(np.nansum(eval_list)),
                    len(eval_list),
                    p=0.5,
                    alternative="greater",
                ).pvalue

                with open(
                    f"data/runs/{run_name}/evaluation/{number}/"
                    f"{pattern.removesuffix('.parquet')}.csv",
                    "w",
                ) as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([eval_str, down_test, up_test])

    print()
    print(
        f"All done. Total evaluation time: {
            round(time.perf_counter() - total_time, 2)
        }s",
        end="\n\n",
    )


def print_status_bar(pattern_name: str, i: int, total_patterns: int) -> None:
    """
    Prints out the status bar (function name, progress bar and count).

    Parameters
    ----------
    pattern_name : str
        Name of the pattern being evaluated.
    i : int
        Current iteration number.
    total_patterns : int
        Total number of patterns.

    Returns
    -------
    None
        Prints a line that overwrites the previous status bar.
    """
    left_line = (51 * (i + 1) // total_patterns) * "-"
    right_line = (51 - len(left_line)) * "-"
    progress_bar = f"|{left_line}>>{right_line}|"
    status_bar = (
        f"Evaluating {pattern_name:<52}"
        + progress_bar
        + f"({i + 1:>3}/{total_patterns})"
    )
    print(status_bar, end="\r")


@numba.jit
def find_first_breakthrough(
    HL_array: np.ndarray, OP: float, open_index: int, limit: int, percent: float
) -> float:
    if OP < 0:
        percent = -percent
    for idx in range(open_index, limit):
        if HL_array[idx, 0] >= OP * (1 + percent / 100) and HL_array[idx, 1] <= OP * (
            1 - percent / 100
        ):
            return np.nan
        if HL_array[idx, 0] >= OP * (1 + percent / 100):
            return 1
        if HL_array[idx, 1] <= OP * (1 - percent / 100):
            return 0
    return np.nan


def n_holding_periods(df_single_close: pd.DataFrame) -> None:
    del df_single_close["high"], df_single_close["low"], df_single_close["volume"]
    df_single_close = df_single_close.rename(columns={"close": "close_0"})
    shifted_closes = {
        f"close_{n}": df_single_close["close_0"].shift(-n) for n in range(1, 10)
    }
    df_all_closes = pd.concat(
        [df_single_close, pd.DataFrame(shifted_closes, index=df_single_close.index)],
        axis=1,
    )

    total_time = time.perf_counter()

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
        i = 1
        n = len(os.listdir(f"../data/patterns/{number}"))
        t = time.perf_counter()

        for pattern in os.listdir(f"../data/patterns/{number}"):
            print(
                f"Evaluating {pattern:<54} | {'#' * (50 * i // n):<50} ({i:>3}/{n})",
                end="\r",
            )

            if (
                pyarrow.parquet.read_table(f"../data/patterns/{number}/{pattern}")
                .to_pandas()
                .sum()
                .to_numpy()[0]
                == 0
            ):
                buy_eval = [("/", "/") for _ in range(10)]
                sell_eval = [("/", "/") for _ in range(10)]

            else:
                df_all_closes["pat"] = (
                    pyarrow.parquet.read_table(f"../data/patterns/{number}/{pattern}")
                    .to_pandas()
                    .set_index(df_all_closes.index)
                    .shift(1)
                )

                subset = df_all_closes[df_all_closes["pat"]]

                mean_buy_profit = {
                    f"{n + 1}_holding_periods": np.format_float_positional(
                        (subset[f"close_{n}"] - subset["open"]).mean(), 2
                    )
                    for n in range(10)
                }
                buy_winning_rate = {
                    f"{n + 1}_holding_periods": np.format_float_positional(
                        100
                        * (subset[f"close_{n}"] > subset["open"]).sum()
                        / len(subset),
                        2,
                    )
                    for n in range(10)
                }
                mean_sell_profit = {
                    f"{n + 1}_holding_periods": np.format_float_positional(
                        (subset["open"] - subset[f"close_{n}"]).mean(), 2
                    )
                    for n in range(10)
                }
                sell_winning_rate = {
                    f"{n + 1}_holding_periods": np.format_float_positional(
                        100
                        * (subset["open"] > subset[f"close_{n}"]).sum()
                        / len(subset),
                        2,
                    )
                    for n in range(10)
                }

                buy_eval = [
                    (mean_buy_profit[key], buy_winning_rate[key])
                    for key in mean_buy_profit
                ]
                sell_eval = [
                    (mean_sell_profit[key], sell_winning_rate[key])
                    for key in mean_sell_profit
                ]

            pyarrow.parquet.write_table(
                pyarrow.table({"buy": buy_eval, "sell": sell_eval}),
                f"../data/evaluation/{number}/{pattern}",
                compression="LZ4",
            )

            i += 1

        print()
        print(
            f"Evaluating patterns with {number} candlestick(s): "
            f"Done in {round(time.perf_counter() - t, 2):<3.2f}s.",
            end="\n\n",
        )

    print(
        f"All done. Total evaluation time: {
            round(time.perf_counter() - total_time, 2)
        }s",
        end="\n\n",
    )
