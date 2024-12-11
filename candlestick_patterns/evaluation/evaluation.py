import os
import time

import numba
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import binomtest, ttest_1samp


def stop_loss_take_profit_evaluation(df: pd.DataFrame) -> None:
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
        Win %, average profit, "less", "greater" and "two-sided" binomial tests to disk.
    """
    del df["close"], df["trend"], df["volume"]

    HL_array = df[["high", "low"]].to_numpy()
    nticks = 40

    total_time = time.time()

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
        t = time.time()

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

            n_detected = (
                pq.read_table(f"../data/patterns/{number}/{pattern}")
                .to_pandas()
                .sum()
                .values[0]
            )
            if n_detected == 0 or n_detected == 1:
                pa.parquet.write_table(
                    pa.table(
                        {
                            "evaluation": "/",
                            "uptest": "/",
                            "downtest": "/",
                            "bothtest": "/",
                        }
                    ),
                    f"../data/evaluation/{number}/{pattern}",
                    compression="LZ4",
                )

            else:

                df["pat"] = (
                    pq.read_table(f"../data/patterns/{number}/{pattern}")
                    .to_pandas()
                    .set_index(df.index)
                    .shift(1)
                )

                patternidxs = df[df["pat"] == True].index
                del df["pat"]

                evallist = []

                for patidx in patternidxs:
                    openidx = df.index.get_loc(patidx)
                    O = df.loc[patidx, "open"]

                    evallist.append(
                        find_first_breakthrough(HL_array, O, openidx, len(df), nticks)
                    )

                evallist = np.array(evallist)
                evalstr = [
                    str(
                        [
                            f"{round(100*np.nansum(evallist)/len(evallist),2):>2.2f}%",
                            f"${round(
                            0.25
                            * nticks
                            * (2 * np.nansum(evallist) - len(evallist))
                            / len(evallist),
                            2,
                        ):>2.2f}",
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
                bothtest = round(
                    binomtest(
                        int(np.nansum(evallist)),
                        len(evallist),
                        p=0.5,
                        alternative="two-sided",
                    ).pvalue,
                    6,
                )
                bothtest = [f"{bothtest} (*)" if bothtest < 0.05 else str(bothtest)]

                pa.parquet.write_table(
                    pa.table(
                        {
                            "evaluation": evalstr,
                            "uptest": uptest,
                            "downtest": downtest,
                            "bothtest": bothtest,
                        }
                    ),
                    f"../data/evaluation/{number}/{pattern}",
                    compression="LZ4",
                )

            i += 1

        print()
        print(
            f"Evaluating patterns with {number} candlestick(s): "
            + f"Done in {round(time.time()-t,2):<3.2f}s.",
            end="\n\n",
        )

    print(
        f"All done. Total evaluation time: {round(time.time()-total_time,2)}s",
        end="\n\n",
    )


@numba.jit
def find_first_breakthrough(HL_array, O, openidx, limit, nticks):
    for idx in range(openidx, limit):
        if (
            HL_array[idx, 0] >= O + 0.25 * nticks
            and HL_array[idx, 1] <= O - 0.25 * nticks
        ):
            return np.nan
        if HL_array[idx, 0] >= O + 0.25 * nticks:
            return 1
        if HL_array[idx, 1] <= O - 0.25 * nticks:
            return 0
    return np.nan


def n_holding_periods(df):
    del df["high"], df["low"], df["volume"]
    df = df.rename(columns={"close": "close_0"})
    shifted_closes = {f"close_{n}": df["close_0"].shift(-n) for n in range(1, 10)}
    df = pd.concat([df, pd.DataFrame(shifted_closes, index=df.index)], axis=1)

    total_time = time.time()

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
        t = time.time()

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
            + f"Done in {round(time.time()-t,2):<3.2f}s.",
            end="\n\n",
        )

    print(
        f"All done. Total evaluation time: {round(time.time()-total_time,2)}s",
        end="\n\n",
    )
