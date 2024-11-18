from prettytable import PrettyTable
import os
import pyarrow.parquet as pq
import time
from word2number import w2n


def run():
    t = time.time()
    print("Making summary table", end="\r")

    table = PrettyTable()
    table.field_names = ["Pattern", "Number of candlesticks", "Number detected"]
    table.align["Pattern"] = "l"
    table.align["Number detected"] = "r"

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
        for pattern in os.listdir(f"../Data/Patterns/{number}"):
            table.add_row(
                [
                    f"{pattern.removesuffix(".parquet").replace("_"," ")}",
                    w2n.word_to_num(number),
                    pq.read_table(f"../Data/Patterns/{number}/{pattern}")
                    .to_pandas()
                    .sum()
                    .values[0],
                ],
                divider=(pattern == os.listdir(f"../Data/Patterns/{number}")[-1]),
            )

    with open("summary.txt", "w") as f:
        f.write(str(table))

    print(f"Making summary table done in {round(time.time()-t,2):>3.2f}s", end="\n\n")


if __name__ == "__main__":
    run()
