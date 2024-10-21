"""Script that handles the input data (CSV in `.txt` format).
Performs some elementary operations and outputs a `.parquet` file,
much smaller than the input.
"""

import pandas as pd
import chrono

# Read data
FILE = "ES Continuous Contract"
df = pd.read_csv(f"Data/{FILE}.txt")
for col in df:
    df[col] = df[col].astype("category")

# Convert date from MM/DD/YYYY to YYYY-MM-DD
# Combine date and time into datetime, which is useful as unique index
df["date"] = pd.to_datetime(df["date"]).astype(str)
df["datetime"] = df.apply(lambda x: chrono.combinedateandtime(x.date, x.time), axis=1)
df = df.drop(columns=["date", "time"])

# Output to .parquet
# lz4 is benchmarked to have the fastest (de)compression speeds, if the files take up
# too much space, brotli provides a better compression ratio at the cost of speed
df.to_parquet("Data/ESCC.parquet", compression="lz4")
