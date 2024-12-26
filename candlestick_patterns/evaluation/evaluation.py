import os
import time

import numba
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import binomtest
from word2number import w2n

pd.set_option("future.no_silent_downcasting", True)


def stop_loss_take_profit_evaluation(df: pd.DataFrame, mode: str = "exclude") -> None:
    """
    Stop loss/take profit-based candlestick pattern evaluation.

    For every time a pattern is detected, checks which margin is triggered first:
    the stop loss or the take profit one.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame with OHLC data. Candlestick patterns are read from disk.
    mode : {"exclude", "ignore", "only"}
        Mode of handling gaps in the data.

    Returns
    -------
    None
        Win %, "less" and "greater" binomial tests to disk.
    """
    del df["close"], df["trend"], df["volume"]

    HL_ARRAY = df[["high", "low"]].to_numpy()
    percent = 1

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
        n = len(os.listdir(f"data/patterns/{number}"))
        t = time.perf_counter()

        for i, pattern in enumerate(os.listdir(f"data/patterns/{number}")):

            try:
                os.remove(f"data/evaluation/{number}/{pattern}")
            except FileNotFoundError:
                pass

            print(
                f"Evaluating {pattern:<54} | "
                + f"{'#'*(50*(i+1)//n):<50} "
                + f"({i+1:>3}/{n})",
                end="\r",
            )

            df["pat"] = (
                pq.read_table(f"data/patterns/{number}/{pattern}")
                .to_pandas()
                .set_index(df.index)
                .shift(1)
            )
            df.iat[0, 4] = False
            num_detected = df["pat"].sum()

            if num_detected == 0 or num_detected == 1:
                pa.parquet.write_table(
                    pa.table(
                        {
                            "evaluation": "/",
                            "uptest": "/",
                            "downtest": "/",
                            "bothtest": "/",
                        }
                    ),
                    f"data/evaluation/{number}/{pattern}",
                    compression="LZ4",
                )

            else:
                patternidxs = df[df["pat"] == True].index
                del df["pat"]

                evallist = np.array([])

                for patidx in patternidxs:
                    openidx = df.index.get_loc(patidx)
                    O = df.loc[patidx, "open"]

                    evallist = np.append(
                        evallist,
                        find_first_breakthrough(HL_ARRAY, O, openidx, len(df), percent),
                    )

                evalstr = [
                    str(
                        [
                            f"{round(100*np.nansum(evallist)/len(evallist),2):>2.2f}%",
                        ]
                    )
                ]
                uptest = round(
                    binomtest(
                        int(np.nansum(evallist)),
                        len(evallist),
                        p=0.5,
                        alternative="greater",
                    ).pvalue,
                    6,
                )
                uptest = [f"{uptest} (*)" if uptest < 0.05 else str(uptest)]
                downtest = round(
                    binomtest(
                        int(np.nansum(evallist)),
                        len(evallist),
                        p=0.5,
                        alternative="less",
                    ).pvalue,
                    6,
                )
                downtest = [f"{downtest} (*)" if downtest < 0.05 else str(downtest)]

                pa.parquet.write_table(
                    pa.table(
                        {"evaluation": evalstr, "uptest": uptest, "downtest": downtest}
                    ),
                    f"data/evaluation/{number}/{pattern}",
                    compression="LZ4",
                )

        print()
        print(
            f"Evaluating patterns with {number} candlestick(s): "
            + f"Done in {round(time.perf_counter()-t,2):<3.2f}s.",
            end="\n\n",
        )

    print(
        f"All done. Total evaluation time: {round(time.perf_counter()-total_time,2)}s",
        end="\n\n",
    )


@numba.jit
def find_first_breakthrough(HL_array, O, openidx, limit, percent):
    for idx in range(openidx, limit):
        if HL_array[idx, 0] >= O * (1 + percent / 100) and HL_array[idx, 1] <= O * (
            1 - percent / 100
        ):
            return np.nan
        if HL_array[idx, 0] >= O * (1 + percent / 100):
            return 1
        if HL_array[idx, 1] <= O * (1 - percent / 100):
            return 0
    return np.nan


def n_holding_periods(df):
    del df["high"], df["low"], df["volume"]
    df = df.rename(columns={"close": "close_0"})
    shifted_closes = {f"close_{n}": df["close_0"].shift(-n) for n in range(1, 10)}
    df = pd.concat([df, pd.DataFrame(shifted_closes, index=df.index)], axis=1)

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

            try:
                os.remove(f"../data/evaluation/{number}/{pattern}")
            except FileNotFoundError:
                pass

            print(
                f"Evaluating {pattern:<54} | "
                + f"{'#'*(50*i//n):<50} "
                + f"({i:>3}/{n})",
                end="\r",
            )

            if (
                pq.read_table(f"../data/patterns/{number}/{pattern}")
                .to_pandas()
                .sum()
                .values[0]
                == 0
            ):
                buyeval = list(("/", "/") for _ in range(10))
                selleval = list(("/", "/") for _ in range(10))

            else:

                df["pat"] = (
                    pq.read_table(f"../data/patterns/{number}/{pattern}")
                    .to_pandas()
                    .set_index(df.index)
                    .shift(1)
                )

                subset = df[df["pat"] == True]

                mean_buy_profit = {
                    f"{n+1}_holding_periods": np.format_float_positional(
                        (subset[f"close_{n}"] - subset["open"]).mean(), 2
                    )
                    for n in range(10)
                }
                buy_winning_rate = {
                    f"{n+1}_holding_periods": np.format_float_positional(
                        100
                        * (subset[f"close_{n}"] > subset["open"]).sum()
                        / len(subset),
                        2,
                    )
                    for n in range(10)
                }
                mean_sell_profit = {
                    f"{n+1}_holding_periods": np.format_float_positional(
                        (subset["open"] - subset[f"close_{n}"]).mean(), 2
                    )
                    for n in range(10)
                }
                sell_winning_rate = {
                    f"{n+1}_holding_periods": np.format_float_positional(
                        100
                        * (subset["open"] > subset[f"close_{n}"]).sum()
                        / len(subset),
                        2,
                    )
                    for n in range(10)
                }

                buyeval = list(
                    (mean_buy_profit[key], buy_winning_rate[key])
                    for key in mean_buy_profit.keys()
                )
                selleval = list(
                    (mean_sell_profit[key], sell_winning_rate[key])
                    for key in mean_sell_profit.keys()
                )

            pa.parquet.write_table(
                pa.table({"buy": buyeval, "sell": selleval}),
                f"../data/evaluation/{number}/{pattern}",
                compression="LZ4",
            )

            i += 1

        print()
        print(
            f"Evaluating patterns with {number} candlestick(s): "
            + f"Done in {round(time.perf_counter()-t,2):<3.2f}s.",
            end="\n\n",
        )

    print(
        f"All done. Total evaluation time: {round(time.perf_counter()-total_time,2)}s",
        end="\n\n",
    )
