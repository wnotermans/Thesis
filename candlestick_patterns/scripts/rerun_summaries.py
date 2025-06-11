import os
import time

from shared import shared_functions
from summary import summary_table


def rerun_all_summaries() -> None:
    """Script that reruns summaries of all runs present in ``src/data/runs/``."""
    os.chdir("src")
    run_names = os.listdir("data/runs")

    N = len(run_names)
    t = time.perf_counter()
    print()
    for i, run_name in enumerate(run_names):
        shared_functions.print_status_bar("", i, N, 5)
        summary_table.make_summaries(run_name=run_name)
        print("\033[F\033[F")
    shared_functions.print_status_bar(
        f"Rerunning summary tables done in {time.perf_counter() - t:3.2f}s", N - 1, N, 5
    )
    print()


if __name__ == "__main__":
    rerun_all_summaries()
