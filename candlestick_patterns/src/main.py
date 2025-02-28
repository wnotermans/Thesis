import time

from detection import pattern_detection
from evaluation import evaluation
from reading import read_data
from summary import summary_table
from trend import trend_calculation


def main():
    t = time.perf_counter()
    with open("logo.txt", encoding="utf-8") as f:
        print(f.read(), end="\n\n")
    DATA = ["Wiener_small"]
    for data in DATA:
        for interval_minutes in [1]:
            print(" Reading data ".center(127, "#"), end="\n\n")
            df, percentiles = read_data.read_and_preprocess(data, interval_minutes)
            print(" Calculating trend ".center(127, "#"), end="\n\n")
            df = trend_calculation.calculate_trend(df)
            print(" Pattern detection ".center(127, "#"), end="\n\n")
            pattern_detection.detection(df, percentiles)
            print(" Pattern evaluation ".center(127, "#"), end="\n\n")
            evaluation.stop_loss_take_profit_evaluation(df)
            print(" Summary table ".center(127, "#"), end="\n\n")
            summary_table.make_summary(f"{data}_{interval_minutes}_minutes")
    print(f" All done in {round(time.perf_counter() - t, 2):>3.2f}s ".center(127, "#"))


if __name__ == "__main__":
    main()
