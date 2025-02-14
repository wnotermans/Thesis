import time

from detection import pattern_detection
from evaluation import evaluation
from reading import read_data
from summary import summary_table


def main():
    t = time.perf_counter()
    with open("logo.txt", "r", encoding="utf-8") as f:
        print(f.read())
    DATA = "Wiener"
    for interval_minutes in [10]:
        # print(" Reading data ".center(127, "#"), end="\n\n")
        # df, percentiles = read_data.read_and_preprocess(DATA, interval_minutes)
        # print(" Pattern detection ".center(127, "#"), end="\n\n")
        # pattern_detection.detection(df, percentiles)
        # print(" Pattern evaluation ".center(127, "#"), end="\n\n")
        # evaluation.stop_loss_take_profit_evaluation(df)
        print(" Summary table ".center(127, "#"), end="\n\n")
        summary_table.make_summary(f"summary_{DATA}_{interval_minutes}_minutes.txt")
    print(f" All done in {round(time.perf_counter()-t,2):>3.2f}s ".center(127, "#"))


if __name__ == "__main__":
    main()
