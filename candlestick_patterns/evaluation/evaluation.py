import os
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def run(df):
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

            print(
                f"Evaluating {pattern:<54} | "
                + f"{'#'*(50*i//n):<50} "
                + f"({i:>3}/{n})",
                end="\r",
            )

            df["pat"] = (
                pq.read_table(f"../data/patterns/{number}/{pattern}")
                .to_pandas()
                .set_index(df.index)
                .shift(1)
            )

            subset = df[df["pat"] == True]
            if len(subset) == 0:
                i += 1
                continue

            mean_buy_profit = {
                f"{n+1}_holding_periods": np.format_float_positional(
                    (subset[f"close_{n}"] - subset["open"]).mean(), 2
                )
                for n in range(10)
            }
            buy_winning_rate = {
                f"{n+1}_holding_periods": np.format_float_positional(
                    100 * (subset[f"close_{n}"] > subset["open"]).sum() / len(subset), 2
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
                    100 * (subset["open"] > subset[f"close_{n}"]).sum() / len(subset),
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
