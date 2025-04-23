import csv
import os
import time

import numba
import numpy as np
import pandas as pd
import pyarrow.parquet
from scipy.stats import binomtest

from shared import constants, shared_functions


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

    high_low_array = df[["high", "low"]].to_numpy()

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
            shared_functions.print_status_bar(
                pattern, i, constants.TOTAL_NUMBER_OF_PATTERNS
            )

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

                eval_list = np.array([])

                for pattern_index in pattern_indices:
                    open_index = df.index.get_loc(pattern_index)
                    OP = df.loc[pattern_index, "open"]

                    eval_list = np.append(
                        eval_list,
                        find_first_breakthrough(
                            high_low_array,
                            OP,
                            open_index,
                            len(df),
                            constants.STOP_LOSS_MARGIN_PERCENT,
                        ),
                    )
                df["evaluation"] = np.nan
                df.loc[df["pat"], "evaluation"] = eval_list
                del df["pat"]

                wins = int(np.nansum(eval_list))
                total = len(eval_list)
                win_rate = wins / total
                absolute_win_rate = (
                    f"{win_rate:.2%}+"
                    if win_rate >= 0.5  # noqa: PLR2004
                    else f"{0.5 + abs(0.5 - win_rate):.2%}-"
                )
                down_test = binomtest(
                    wins,
                    total,
                    p=0.5,
                    alternative="less",
                ).pvalue
                up_test = binomtest(
                    wins,
                    total,
                    p=0.5,
                    alternative="greater",
                ).pvalue

                with open(
                    f"data/runs/{run_name}/evaluation/{number}/"
                    f"{pattern.removesuffix('.parquet')}.csv",
                    "w",
                ) as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([absolute_win_rate, down_test, up_test])


@numba.jit
def find_first_breakthrough(
    HL_array: np.ndarray, OP: float, open_index: int, limit: int, percent: float
) -> float:
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


def n_holding_periods(df_single_close: pd.DataFrame, *, run_name: str) -> None:
    del df_single_close["high"], df_single_close["low"], df_single_close["volume"]
    df_single_close = df_single_close.rename(columns={"close": "close_0"})
    shifted_closes = {
        f"close_{n}": df_single_close["close_0"].shift(-n) for n in range(1, 10)
    }
    df_all_closes = pd.concat(
        [df_single_close, pd.DataFrame(shifted_closes, index=df_single_close.index)],
        axis=1,
    )

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
        n = len(os.listdir(f"data/runs/{run_name}/detection/{number}"))

        for pattern in os.listdir(f"data/runs/{run_name}/detection/{number}"):
            print(
                f"Evaluating {pattern:<54} | {'#' * (50 * i // n):<50} ({i:>3}/{n})",
                end="\r",
            )

            if (
                pyarrow.parquet.read_table(
                    f"data/runs/{run_name}/detection/{number}/{pattern}"
                )
                .to_pandas()
                .sum()
                .to_numpy()[0]
                == 0
            ):
                buy_eval = [("/", "/") for _ in range(10)]
                sell_eval = [("/", "/") for _ in range(10)]

            else:
                df_all_closes["pat"] = (
                    pyarrow.parquet.read_table(
                        f"data/runs/{run_name}/detection/{number}/{pattern}"
                    )
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
                f"data/runs/{run_name}/evaluation/{number}/{pattern}",
                compression="LZ4",
            )

            i += 1


EVALUATION_METHODS = {
    "stop_loss_take_profit": stop_loss_take_profit_evaluation,
    "n_holding_periods": n_holding_periods,
}


def evaluation(df: pd.DataFrame, evaluation_method: str, *, run_name: str) -> None:
    """
    Performs evaluation of the patterns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to be analyzed
    evaluation_method : str
        Which evaluation method to use.
    """
    t = time.perf_counter()
    evaluation_func = EVALUATION_METHODS[evaluation_method]
    evaluation_func(df, run_name=run_name)
    print()
    print(
        f"All done. Total evaluation time: {time.perf_counter() - t:3.2f}s",
        end="\n\n",
    )
