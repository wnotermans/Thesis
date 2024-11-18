import mplfinance as mpf
import pandas as pd
import numpy as np
import candlestick_functions as cf
import patterns as pat
import matplotlib.pyplot as plt


df = pd.read_parquet("../Data/ESCC.parquet")  # read data
df["datetime"] = pd.to_datetime(df["datetime"])  # parse datetime
df = df.set_index("datetime")  # set datetime as index for mplfinance

df["body"] = df.apply(lambda x: cf.hb(x[0], x[3]), axis=1, raw=True, engine="numba")
# df["us"] = df.apply(
#     lambda x: cf.upper_shadow(x[0], x[1], x[3]), axis=1, raw=True, engine="numba"
# )
# df["ls"] = df.apply(
#     lambda x: cf.lower_shadow(x[0], x[2], x[3]), axis=1, raw=True, engine="numba"
# )

# print(len(df[df["us"] == 0]))
# print(len(df[df["us"] == 0.25]))
# print(len(df[df["us"] == 0.5]))
# print(df[df["body"] != 0].head())
print(np.quantile(df[df["body"] != 0]["body"], q=(0.25, 0.5, 0.75)))
