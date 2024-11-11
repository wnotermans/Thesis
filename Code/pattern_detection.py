import pandas as pd
import numpy as np
import pyarrow as pa
import candlestick_functions as cf
import one_patterns
import two_patterns
import time


def run():

    total_time = time.time()

    print("Reading and handling data", end="\r")
    t = time.time()
    df = pd.read_parquet("../Data/ESCC.parquet")
    df["datetime"] = pd.to_datetime(df["datetime"])  # parse datetime
    df = df.set_index("datetime")  # set datetime as index for mplfinance
    print(f"Reading and handling data done in {round(time.time()-t,2):<3.2f}s")

    print("Calculating moving average", end="\r")
    t = time.time()
    df["5_MA"] = df["close"].rolling(5).mean()
    print(f"Calculating moving average done in {round(time.time()-t,2):<3.2f}s")

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
    print(f"Calculating trend done in {round(time.time()-t,2):<3.2f}s")

    column_dict = {
        f"{col}_{shift}": df[col].shift(shift).values
        for col in ["open", "high", "low", "close", "volume"]
        for shift in range(5)
    }

    candle_dict = {
        f"candle_minus_{n}": np.array(
            [
                column_dict[f"open_{n}"],
                column_dict[f"high_{n}"],
                column_dict[f"low_{n}"],
                column_dict[f"close_{n}"],
                column_dict[f"volume_{n}"],
            ]
        ).T
        for n in range(5)
    }

    T = np.array(df["5_MA_trend"].values)

    one_candle = candle_dict["candle_minus_0"]
    two_candle = [candle_dict[f"candle_minus_{n}"] for n in range(1, -1, -1)]

    for number in ["one", "two"]:
        print(f"Candlestick patterns with {number} candlestick(s)")

        pattern_funcs = []  # get all functions names of the pattern functions
        for name in dir(globals().get(f"{number}_patterns")):
            attr = getattr(globals().get(f"{number}_patterns"), name)
            if callable(attr):
                pattern_funcs.append(name)

        i = 1  # counter for total number of patterns
        num_funcs = len(pattern_funcs)

        t = time.time()
        for func_name in pattern_funcs:
            print(
                f"Detecting pattern {func_name:>50} | "
                + f"{'#'*(50*i//num_funcs):<50} "
                + f"({i:>3}/{num_funcs})",
                end="\r",
            )
            func_call = getattr(globals().get(f"{number}_patterns"), func_name)
            pat = np.zeros(len(df), dtype=bool)
            pat = func_call(locals().get(f"{number}_candle"), T)
            pa.parquet.write_table(
                pa.table({f"{func_name}": pat}),
                f"../Data/Patterns/{number}/{func_name}.parquet",
                compression="LZ4",
            )
            i += 1
        print()
        print(
            f"Detecting patterns with {number} candlestick(s): "
            + f"Done in {round(time.time()-t,2):<3.2f}s."
        )
    print(f"All done. Total detection time: {round(time.time()-total_time,2)}s")

    if __name__ == "__main__":
        run()
