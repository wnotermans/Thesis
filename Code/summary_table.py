from prettytable import PrettyTable
import os
import pyarrow.parquet as pq


def run():
    table = PrettyTable()
    table.field_names = ["Pattern", "Number of candlesticks", "Number detected"]
    table.align["Pattern"] = "l"
    table.align["Number detected"] = "r"
    i = 1
    for number in ["one", "two"]:
        for pattern in os.listdir(f"../Data/Patterns/{number}"):
            table.add_row(
                [
                    f"{pattern.strip(".parquet").replace("_"," ")}",
                    i,
                    pq.read_table(f"../Data/Patterns/{number}/{pattern}")
                    .to_pandas()
                    .to_numpy()
                    .sum(),
                ],
                divider=(pattern == os.listdir(f"../Data/Patterns/{number}")[-1]),
            )
        i += 1
    print(table)
    with open("summary.txt", "w") as f:
        f.write(str(table))


if __name__ == "__main__":
    run()
