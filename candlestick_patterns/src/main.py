import time

from detection import pattern_detection
from evaluation import evaluation
from folder_setup import folder_setup
from reading import read_data
from summary import summary_table
from trend import trend_calculation


def main() -> None:
    t = time.perf_counter()
    with open("logo.txt", encoding="utf-8") as f:
        print(f.read(), end="\n\n")
    FILENAMES = ["Wiener_small"]
    # FILENAMES = ["BND", "GLD", "QQQ", "SPY", "Wiener"]
    STARTTIME = "09:30:00"
    ENDTIME = "16:00:00"
    FILTER_NEWS = False
    AVERAGING_METHOD = "MA"  # MA, WMA, EWMA
    SPAN = 5
    DECISION_METHOD = "counting"  # monotonic, counting, high_low
    CONSECUTIVE = 7
    DETECTION_METHOD = "stop_loss_take_profit"
    for filename in FILENAMES:
        for interval_minutes in [5, 10]:
            run_name = (
                f"{filename}_{interval_minutes}min_"
                f"{AVERAGING_METHOD}_{DECISION_METHOD}_{DETECTION_METHOD}"
            )
            parameters = {
                "filename": filename,
                "interval_minutes": interval_minutes,
                "start_time": STARTTIME,
                "end_time": ENDTIME,
                "averaging_method": AVERAGING_METHOD,
                "span": SPAN,
                "decision_method": DECISION_METHOD,
                "consecutive": CONSECUTIVE,
                "detection_method": DETECTION_METHOD,
                "filter_news": FILTER_NEWS,
            }
            box_width = max(
                [len(f"{key}={value}") for key, value in parameters.items()]
            )
            print(f"+{'Parameters':-^{box_width}}+")
            for key, value in parameters.items():
                line = "=".join([key, str(value)])
                print(f"|{line:^{box_width}}|")
            print(f"+{'-' * box_width}+")

            mode, unique_id = folder_setup.folder_setup(
                parameters, run_name=run_name, set_mode=None
            )
            run_name = unique_id + "_" + run_name
            if mode == "pass":
                continue
            if mode == "rerun":
                print(" Reading data ".center(127, "#"), end="\n\n")
                main_set, percentiles = read_data.read_and_preprocess(
                    filename,
                    interval_minutes,
                    STARTTIME,
                    ENDTIME,
                    filter_news=FILTER_NEWS,
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
                pattern_detection.detection(
                    main_set_with_trend, percentiles, run_name=run_name
                )

                print(" Pattern evaluation ".center(127, "#"), end="\n\n")
                evaluation.evaluation(
                    main_set_with_trend,
                    detection_method=DETECTION_METHOD,
                    run_name=run_name,
                )

            print(" Summary table ".center(127, "#"), end="\n\n")
            summary_table.make_summary(run_name=run_name)
            print("".center(127, "#"), end="\n\n")
    print(f" All done in {round(time.perf_counter() - t, 2):>3.2f}s ".center(127, "#"))


if __name__ == "__main__":
    main()
