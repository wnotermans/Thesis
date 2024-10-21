import mplfinance as mpf
import pandas as pd
import numpy as np
import candlestick_functions as cf
import patterns as pat
from inspect import getmembers, isfunction
from tqdm import tqdm


def main():
    print("Reading data...")
    df = pd.read_parquet("../Data/ESCC.parquet")[:100000]
    nrow = len(df)
    print("Done.")
    print("Setting date as index...")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime")
    print("Done.")
    print("Calculating moving average...")
    df["5_MA"] = df["close"].rolling(5).mean()
    print("Done.")
    tqdm.pandas(desc="Calculating trend", total=nrow)
    df["5_MA_trend"] = df["5_MA"].rolling(7).progress_apply(cf.trend, raw=True)
    # df["hb"] = df.apply(lambda x: cf.hb(x.open, x.close), axis=1)
    patterns = [x[0] for x in getmembers(pat, isfunction)]
    i = 1
    for pattern in patterns:
        # print(f"Detecting pattern {pattern} ({i}/{len(patterns)})")
        tqdm.pandas(desc=f"Detecting {pattern} ({i}/{len(patterns)})", total=nrow)
        i += 1
        df[pattern] = df.progress_apply(
            lambda x: getattr(pat, pattern)(x.open, x.high, x.low, x.close), axis=1
        )
        subset = df[df[pattern] == True]
        # fig, axlist = mpf.plot(
        #     subset[: min(len(subset), 10)],
        #     type="candle",
        #     volume=True,
        #     returnfig=True,
        # )
        n = len(subset)
        print(f"Detected {pattern} {n} times.")
        # axlist[0].set_title(f"{pattern}, n={n}")
        # mpf.show()


if __name__ == "__main__":
    main()
