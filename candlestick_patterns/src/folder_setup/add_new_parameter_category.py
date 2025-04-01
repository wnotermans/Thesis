import os

NEW_CATEGORY = ""
DEFAULT_VALUE = ""

input(f"Adding {NEW_CATEGORY = } with {DEFAULT_VALUE = }. Enter to confirm.")
for folder in os.listdir("../data/runs"):
    with open(f"../data/runs/{folder}/parameters.txt", "a") as file:
        file.write(f"_{'='.join([NEW_CATEGORY, DEFAULT_VALUE])}")
