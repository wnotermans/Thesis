import os
import uuid

PATTERN_NUMBERS = [
    "one",
    "two",
    "three",
    "four",
    "five",
    "eight",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
]


def folder_setup(
    input_parameters: dict, *, run_name: str, set_mode: int | None = None
) -> tuple[str, str]:
    """
    Setup folders for detection/evaluation run.

    Includes handling of already existing runs and the creation of new folders.

    Parameters
    ----------
    input_parameters : dict
        Dictionary of the detection/evaluation parameters.
    run_name : str
        Folder name of the run. A unique id is prepended.
    set_mode : int | None, optional, default None
        Allows setting a default mode to skip user interaction.

    Returns
    -------
    tuple[str, str]
        Rerun mode and unique id.
    """
    input_parameters_list = [
        f"{key}={value}".replace("_", "") for key, value in input_parameters.items()
    ]
    already_present_parameter_list = []
    for folder in os.listdir("data/runs"):
        with open(f"data/runs/{folder}/parameters.txt") as file:
            already_present_parameter_list.append(file.readline().split("_"))

    if input_parameters_list in already_present_parameter_list:
        old_id = os.listdir("data/runs")[
            already_present_parameter_list.index(input_parameters_list)
        ][:8]
        return (
            handle_existing_run(set_mode=set_mode),
            old_id,
        )

    return make_new_folder(input_parameters_list, run_name=run_name)


def handle_existing_run(*, set_mode: int | None = None) -> str:
    """
    Handle existing runs, ask the user whether to pass, rerun summary or rerun entirely.

    Parameters
    ----------
    set_mode : int | None, optional, default None
        Allows setting a default mode to skip user interaction.

    Returns
    -------
    str
        Rerun mode.
    """
    if set_mode is None:
        mode = int(
            input(
                "Parameter combination already present, how to proceed?\n"
                "0: pass, 1: remake summary, 2: rerun from scratch. "
            )
        )
        print()
    else:
        mode = set_mode
    if mode == 0:
        return "pass"
    if mode == 1:
        return "summary"
    return "rerun"


def make_new_folder(
    input_parameters_list: list[str], *, run_name: str
) -> tuple[str, str]:
    """
    Handles the creation of a new folder with correct sub-folders,
    prepending a unique id to the root folder.
    Also writes the parameters used in the run to the file `parameters.txt`.

    Parameters
    ----------
    input_parameters_list : list[str]
        List of the input parameters to write to `parameters.txt`
    run_name : str
        Folder name of the run. A unique id is prepended.

    Returns
    -------
    tuple[str, str]
        Rerun mode and unique id.
    """
    random_id = str(uuid.uuid4())[:8]
    for parent_folder in ["detection", "evaluation"]:
        for pattern_number in PATTERN_NUMBERS:
            folder_path = (
                f"data/runs/{random_id}_{run_name}/{parent_folder}/{pattern_number}"
            )
            os.makedirs(folder_path)
    with open(f"data/runs/{random_id}_{run_name}/parameters.txt", "w") as file:
        file.write("_".join(input_parameters_list))
    return ("rerun", random_id)
