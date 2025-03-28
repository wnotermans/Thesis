import os

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


def folder_setup(run_name: str) -> None:
    for parent_folder in ["detection", "evaluation"]:
        for pattern_number in PATTERN_NUMBERS:
            folder_path = f"data/runs/{run_name}/{parent_folder}/{pattern_number}"
            if os.path.exists(folder_path):
                for file in os.listdir(folder_path):
                    os.remove(f"{folder_path}/{file}")
                os.rmdir(folder_path)
            os.makedirs(folder_path)
