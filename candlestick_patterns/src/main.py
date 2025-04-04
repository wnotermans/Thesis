import time

from detection import pattern_detection
from evaluation import evaluation
from folder_setup import folder_setup
from reading import read_data
from shared import constants, shared_functions
from summary import summary_table
from trend import trend_calculation


def main() -> None:
    t = time.perf_counter()
    with open("logo.txt", encoding="utf-8") as f:
        print(f.read(), end="\n\n")
    for filename in constants.FILENAMES:
        for interval_minutes in constants.INTERVAL_MINUTES:
            run_name = (
                f"{filename}_{interval_minutes}min_"
                f"{constants.TREND_AVERAGING_METHOD}_"
                f"{constants.TREND_DECISION_METHOD}_"
                f"{constants.EVALUATION_METHOD}"
            )
            parameters = {
                "filename": filename,
                "interval_minutes": interval_minutes,
                **constants.SHARED_PARAMS_DICT,
            }
            shared_functions.box_print(parameters)

            mode, unique_id = folder_setup.folder_setup(
                parameters, run_name=run_name, set_mode=2
            )
            run_name = unique_id + "_" + run_name
            if mode == "pass":
                continue
            if mode == "rerun":
                print(" Reading data ".center(127, "#"), end="\n\n")
                main_set, percentiles = read_data.read_and_preprocess(
                    filename,
                    interval_minutes,
                    constants.STARTTIME,
                    constants.ENDTIME,
                    filter_news=constants.FILTER_NEWS,
                )

                print(" Calculating trend ".center(127, "#"), end="\n\n")
                main_set_with_trend = trend_calculation.calculate_trend(
                    main_set,
                    averaging_method=constants.TREND_AVERAGING_METHOD,
                    averaging_method_kwargs=constants.TREND_AVERAGING_METHOD_KWARGS,
                    decision_method=constants.TREND_DECISION_METHOD,
                    decision_method_kwargs=constants.TREND_DECISION_METHOD_KWARGS,
                )

                print(" Pattern detection ".center(127, "#"), end="\n\n")
                pattern_detection.detection(
                    main_set_with_trend,
                    percentiles,
                    constants.DATA_GAP_HANDLING,
                    run_name=run_name,
                )

                print(" Pattern evaluation ".center(127, "#"), end="\n\n")
                evaluation.evaluation(
                    main_set_with_trend,
                    evaluation_method=constants.EVALUATION_METHOD,
                    run_name=run_name,
                )

            print(" Summary table ".center(127, "#"), end="\n\n")
            summary_table.make_summary(run_name=run_name)
            print("".center(127, "#"), end="\n\n")
    print(f" All done in {round(time.perf_counter() - t, 2):>3.2f}s ".center(127, "#"))


if __name__ == "__main__":
    main()
