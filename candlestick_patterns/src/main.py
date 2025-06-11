from detection import pattern_detection
from evaluation import cleanup, evaluation
from folder_setup import folder_setup
from indicators import indicators
from reading import read_data
from shared import constants, shared_functions
from summary import summary_table
from trend import trend_calculation


def main() -> None:
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
                main_sets, percentiles_list = read_data.read_and_preprocess(
                    filename,
                    interval_minutes,
                    constants.START_END_TIME,
                    filter_news_kwargs=constants.FILTER_NEWS_KWARGS,
                )

                for n, (main_set, percentiles) in enumerate(
                    zip(main_sets, percentiles_list, strict=False)
                ):
                    main_set_with_trend = trend_calculation.calculate_trend(
                        main_set,
                        averaging_method=constants.TREND_AVERAGING_METHOD,
                        averaging_kwargs=constants.TREND_AVERAGING_METHOD_KWARGS,
                        decision_method=constants.TREND_DECISION_METHOD,
                        decision_kwargs=constants.TREND_DECISION_METHOD_KWARGS,
                    )

                    main_set_with_trend = indicators.calculate_indicators(
                        main_set_with_trend,
                        indicator_kwargs=constants.INDICATOR_KWARGS,
                    )

                    pattern_detection.detection(
                        main_set_with_trend,
                        percentiles,
                        constants.DATA_GAP_HANDLING,
                        run_name=run_name,
                        filter_kwargs=constants.INDICATOR_FILTER_KWARGS,
                        split=n + 1,
                    )

                    evaluation.stop_loss_take_profit_evaluation(
                        main_set_with_trend,
                        constants.STOP_LOSS_TAKE_PROFIT_MARGINS,
                        run_name=run_name,
                        split=n + 1,
                    )
                cleanup.clean(run_name)
                print()

            summary_table.make_summaries(run_name=run_name)


if __name__ == "__main__":
    main()
