import time

from detection import pattern_detection
from evaluation import evaluation
from folder_setup import folder_setup
from indicators import indicators
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
                f"{constants.TREND_DECISION_METHOD}"
            )
            parameters = {
                "filename": filename,
                "interval_minutes": interval_minutes,
                **constants.SHARED_PARAMS_DICT,
            }
            shared_functions.box_print(parameters)

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
                    constants.START_END_TIME,
                    filter_news_kwargs=constants.FILTER_NEWS_KWARGS,
                )

                print(" Calculating trend ".center(127, "#"), end="\n\n")
                main_set_with_trend = trend_calculation.calculate_trend(
                    main_set,
                    averaging_method=constants.TREND_AVERAGING_METHOD,
                    averaging_kwargs=constants.TREND_AVERAGING_METHOD_KWARGS,
                    decision_method=constants.TREND_DECISION_METHOD,
                    decision_kwargs=constants.TREND_DECISION_METHOD_KWARGS,
                )

                print(
                    " Calculating additional indicators ".center(127, "#"), end="\n\n"
                )
                main_set_with_trend = indicators.calculate_indicators(
                    main_set_with_trend,
                    indicator_kwargs=constants.INDICATOR_KWARGS,
                )

                print(" Pattern detection ".center(127, "#"), end="\n\n")
                pattern_detection.detection(
                    main_set_with_trend,
                    percentiles,
                    constants.DATA_GAP_HANDLING,
                    run_name=run_name,
                    filter_kwargs=constants.INDICATOR_FILTER_KWARGS,
                )

                print(" Pattern evaluation ".center(127, "#"), end="\n\n")
                evaluation.stop_loss_take_profit_evaluation(
                    main_set_with_trend,
                    constants.STOP_LOSS_TAKE_PROFIT_MARGINS,
                    run_name=run_name,
                )

            print(" Summary table ".center(127, "#"), end="\n\n")
            summary_table.make_summaries(run_name=run_name)
            print("".center(127, "#"), end="\n\n")
    print(f" All done in {time.perf_counter() - t:3.2f}s ".center(127, "#"))


if __name__ == "__main__":
    main()
