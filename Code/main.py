import mplfinance as mpf
import pandas as pd
import numpy as np
import candlestick_functions as cf
import one_patterns as one_pat
import time


def main():
    total_time = time.time()
    print("Reading and handling data", end="\r")
    t = time.time()
    df = pd.read_parquet("../Data/ESCC.parquet")
    df["datetime"] = pd.to_datetime(df["datetime"])  # parse datetime
    df = df.set_index("datetime")  # set datetime as index for mplfinance
    print(f"Reading and handling data done in {round(time.time()-t,2)}s")

    print("Calculating moving average", end="\r")
    t = time.time()
    df["5_MA"] = df["close"].rolling(5).mean()
    print(f"Calculating moving average done in {round(time.time()-t,2)}s")

    print("Calculating trend", end="\r")
    t = time.time()
    # 7 consecutive increases/decreases in moving average for trend to be defined
    df["5_MA_trend"] = (
        df["5_MA"]
        .rolling(7)
        .apply(
            cf.trend,
            raw=True,
            engine="numba",
        )
    )
    print(f"Calculated trend in {round(time.time()-t,2)}s")

    i = 1
    O = df["open"].values
    H = df["high"].values
    L = df["low"].values
    C = df["close"].values
    V = df["volume"].values
    candle = np.array([O, H, L, C, V]).T
    T = np.array(df["5_MA_trend"].values)

    pattern_funcs = []  # get all functions names of the pattern functions
    for name in dir(one_pat):
        attr = getattr(one_pat, name)
        if callable(attr):
            pattern_funcs.append(name)

    for fun in pattern_funcs:
        print(f"Detecting pattern {fun:<50} ({i:>3}/{len(pattern_funcs)})", end="\r")
        t = time.time()
        func = getattr(one_pat, fun)
        pat = np.zeros(len(df), dtype=bool)
        pat[:] = func(candle, T[:])
        df[fun] = pat

        subset = df[df[fun] == True]
        # fig, axlist = mpf.plot(
        #     subset[: min(len(subset), 10)],
        #     type="candle",
        #     volume=True,
        #     returnfig=True,
        # )
        n = len(subset)
        print(
            f"Detected pattern {fun:<50} ({i:>3}/{len(pattern_funcs)}) {n:<7} time(s) in {round(time.time()-t,2):<3.2f}s."
        )
        df[fun].to_frame(name=fun).to_parquet(
            f"../Data/Patterns/One/{fun}.parquet", compression="lz4", index=False
        )
        del df[fun]
        i += 1
        # axlist[0].set_title(f"{fun}, n={n}")
        # mpf.show()
    print(f"All done. Total running time: {round(time.time()-total_time,2)}s")


if __name__ == "__main__":
    main()
