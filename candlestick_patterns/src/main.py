import time

from detection import pattern_detection
from evaluation import evaluation
from reading import read_data
from summary import summary_table
from trend import trend_calculation


def main() -> None:
    t = time.perf_counter()
    with open("logo.txt", encoding="utf-8") as f:
        print(f.read(), end="\n\n")
    DATA = ["Wiener_small"]
    AVERAGING_METHOD = "EWMA"
    DECISION_METHOD = "counting"
    SPAN = 5
    CONSECUTIVE = 7
    for data in DATA:
        for interval_minutes in [1]:
            print(" Reading data ".center(127, "#"), end="\n\n")
            main_set, percentiles = read_data.read_and_preprocess(
                data, interval_minutes, end_time="10:00:00"
            )
            print(" Calculating trend ".center(127, "#"), end="\n\n")
            main_set_with_trend = trend_calculation.calculate_trend(
                main_set,
                averaging_method=AVERAGING_METHOD,
                span=SPAN,
                decision_method=DECISION_METHOD,
                consecutive=CONSECUTIVE,
            )
            print(" Pattern detection ".center(127, "#"), end="\n\n")
            pattern_detection.detection(main_set_with_trend, percentiles)
            print(" Pattern evaluation ".center(127, "#"), end="\n\n")
            evaluation.stop_loss_take_profit_evaluation(main_set_with_trend)
            print(" Summary table ".center(127, "#"), end="\n\n")
            summary_table.make_summary(
                f"{data=}_{interval_minutes=}_{AVERAGING_METHOD=}_{DECISION_METHOD=}_{SPAN=}_{CONSECUTIVE=}"
            )
    print(f" All done in {round(time.perf_counter() - t, 2):>3.2f}s ".center(127, "#"))


if __name__ == "__main__":
    main()
