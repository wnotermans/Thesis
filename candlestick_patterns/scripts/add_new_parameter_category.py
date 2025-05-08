import os

NEW_CATEGORY = ""
DEFAULT_VALUE = ""


def add_category() -> None:
    """Add new parameter category to ``parameters.txt`` files."""
    input(f"Adding {NEW_CATEGORY = } with {DEFAULT_VALUE = }. Enter to confirm.")
    for folder in os.listdir("src/data/runs"):
        with open(f"src/data/runs/{folder}/parameters.txt", "a") as file:
            file.write(f"#{'='.join([NEW_CATEGORY, DEFAULT_VALUE])}")


if __name__ == "__main__":
    add_category()
