import os
import time
from ast import literal_eval

from detection import pattern_detection
from evaluation import evaluation
from indicators import indicators
from reading import read_data
from shared import shared_functions
from summary import summary_table
from trend import trend_calculation


def rerun_all() -> None:
    os.chdir("src")
    run_names = os.listdir("data/runs")

    N = len(run_names)
    t = time.perf_counter()
    for i, run_name in enumerate(run_names):
        print()
        shared_functions.print_status_bar("Rerunning all", i, N)
        print("\n\n")

        with open(f"data/runs/{run_name}/parameters.txt") as f:
            parameters = dict(keyval.split("=") for keyval in f.read().split("#"))

        main_set, percentiles = read_data.read_and_preprocess(
            parameters["filename"],
            int(parameters["interval_minutes"]),
            literal_eval(parameters["start_end_time"]),
            filter_news_kwargs=literal_eval(parameters["filter_news_kwargs"]),
        )
        main_set_with_trend = trend_calculation.calculate_trend(
            main_set,
            averaging_method=parameters["trend_averaging_method"],
            averaging_kwargs=literal_eval(parameters["trend_averaging_method_kwargs"]),
            decision_method=parameters["trend_decision_method"],
            decision_kwargs=literal_eval(parameters["trend_decision_method_kwargs"]),
        )
        main_set_with_trend = indicators.calculate_indicators(
            main_set_with_trend,
            indicator_kwargs=literal_eval(parameters["indicator_kwargs"]),
        )
        pattern_detection.detection(
            main_set_with_trend,
            percentiles,
            parameters["data_gap_handling"],
            run_name=run_name,
            filter_kwargs=literal_eval(parameters["indicator_filter_kwargs"]),
        )
        evaluation.evaluation(
            main_set_with_trend,
            evaluation_method=parameters["evaluation_method"],
            run_name=run_name,
        )
        summary_table.make_summaries(run_name=run_name)
    print(f"Rerunning all done in {time.perf_counter() - t:3.2f}s")


if __name__ == "__main__":
    rerun_all()
