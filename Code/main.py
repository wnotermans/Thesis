import mplfinance as mpf
import pandas as pd
import numpy as np
import candlestick_functions as cf
import patterns as pat
import time


def main():
    total_time = time.time()
    print("Reading and handling data", end="\r")
    t = time.time()
    df = pd.read_parquet("../Data/ESCC.parquet")  # read data
    df["datetime"] = pd.to_datetime(df["datetime"])  # parse datetime
    df = df.set_index("datetime")  # set datetime as index for mplfinance
    print(f"Reading and handling data done in {round(time.time()-t,1)}s")

    print("Calculating moving average", end="\r")
    t = time.time()
    df["5_MA"] = df["close"].rolling(5).mean()
    print(f"Calculating moving average done in {round(time.time()-t,1)}s")

    print("Calculating trend", end="\r")
    t = time.time()
    df["5_MA_trend"] = (
        df["5_MA"]
        .rolling(7)
        .apply(
            cf.trend,
            raw=True,
            engine="numba",
        )
    )
    print(f"Calculated trend in {round(time.time()-t,1)}s")

    # df["hb"] = df.apply(lambda x: cf.hb(x.open, x.close), axis=1)
    i = 1

    pattern_funcs = []  # get all functions names of the pattern functions
    for name in dir(pat):
        attr = getattr(pat, name)
        if callable(attr):
            pattern_funcs.append(name)

    for fun in pattern_funcs:
        print(f"Detecting pattern {fun:<50} ({i:>3}/{len(pattern_funcs)})", end="\r")
        t = time.time()
        func = getattr(pat, fun)
        df[fun] = df.apply(
            lambda x: func(x[0], x[1], x[2], x[3], x[4], x[6]),
            axis=1,
            raw=True,
            engine="numba",
        )

        subset = df[df[fun] == True]
        # fig, axlist = mpf.plot(
        #     subset[: min(len(subset), 10)],
        #     type="candle",
        #     volume=True,
        #     returnfig=True,
        # )
        n = len(subset)
        print(
            f"Detected pattern {fun:<50} ({i:>3}/{len(pattern_funcs)}) {n:<6} time(s) in {round(time.time()-t,1):<3}s."
        )
        i += 1
        # axlist[0].set_title(f"{fun}, n={n}")
        # mpf.show()
    print(f"All done. Total running time: {round(time.time()-total_time,1)}s")


if __name__ == "__main__":
    main()
