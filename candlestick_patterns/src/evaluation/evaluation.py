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
    open_array = df["open"].to_numpy()
    high_array = df["high"].to_numpy()
    low_array = df["low"].to_numpy()

    indicator_columns = df.columns.difference(
        ["open", "high", "low", "close", "volume", "gap", "trend"]
    )

    i = 0
    for number_str in constants.PATTERN_NUMBERS_AS_STRING:
        for pattern in os.listdir(f"data/runs/{run_name}/detection/{number_str}"):
            shared_functions.print_status_bar(
                pattern, i, constants.TOTAL_NUMBER_OF_PATTERNS
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

            if num_detected in (0, 1):
                with open(
                    f"data/runs/{run_name}/evaluation/{number_str}/"
                    f"{pattern.removesuffix('.parquet')}.csv",
                    "w",
                ) as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["/"] * 3)

            else:
                bool_array = df["pattern"].astype(bool).to_numpy()
                eval_list = find_first_breakthroughs(
                    bool_array,
                    open_array,
                    high_array,
                    low_array,
                    constants.STOP_LOSS_MARGIN_PERCENT,
                )

                df["evaluation"] = None
                df.loc[df["pattern"], "evaluation"] = eval_list
                del df["pattern"]

                success_indicator_means = df.loc[
                    df["evaluation"] == 1, indicator_columns
                ].mean()

                wins = int(np.nansum(eval_list))
                number_detected = len(eval_list)
                win_rate = wins / number_detected
                absolute_win_rate = (
                    f"{win_rate:.2%}+"
                    if win_rate >= 0.5  # noqa: PLR2004
                    else f"{0.5 + abs(0.5 - win_rate):.2%}-"
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

                csv_path = (
                    f"data/runs/{run_name}/evaluation/{number_str}/"
                    f"{pattern.removesuffix('.parquet')}"
                )
                csv_data = {
                    f"{csv_path}evaluation.csv": [
                        absolute_win_rate,
                        down_test,
                        up_test,
                        number_detected,
                    ],
                    f"{csv_path}indicators.csv": success_indicator_means,
                }
                for path, data in csv_data.items():
                    with open(path, "w") as csvfile:
                        csv.writer(csvfile).writerow(data)


@numba.jit
def find_first_breakthroughs(
    bool_array: np.ndarray,
    open_array: np.ndarray,
    high_array: np.ndarray,
    low_array: np.ndarray,
    percentage: float,
) -> list:
    """
    Find the stop loss take profit breakthroughs. Margins are percentage-based and
    controlled globally.

    Parameters
    ----------
    bool_array : np.ndarray
        Boolean array of the detected patterns.
    open_array : np.ndarray
        Opens.
    high_array : np.ndarray
        Highs.
    low_array : np.ndarray
        Lows.
    percentage : float
        Percentage that defines the margins.

    Returns
    -------
    list
        Evaluation: 0 for bottom breakthrough, 1 for upper, ``np.nan`` else.
    """
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
                df_all_closes["pattern"] = (
                    pyarrow.parquet.read_table(
                        f"data/runs/{run_name}/detection/{number}/{pattern}"
                    )
                    .to_pandas()
                    .set_index(df_all_closes.index)
                    .shift(1)
                )

                subset = df_all_closes[df_all_closes["pattern"]]

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
