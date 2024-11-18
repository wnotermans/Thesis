"""Module that provides fuctions that handle various time-related operations and computations.

WARNING: Some (many?) of these only work on single elements, be sure to check the docstring of the specific function.
"""

import datetime


def datetoiso(in_date: str) -> str:
    """Converts date from the given MM/DD/YYYY format to `ISO 8601`'s YYYY-MM-DD format.

    WARNING: Only works on single elements, not on entire columns, so use the `.apply()` method.
    """
    return datetime.datetime.strptime(in_date, "%m/%d/%Y").strftime("%Y-%m-%d")


def datetoordinal(in_date: str) -> int:
    """Converts date from the given MM/DD/YYYY format to ordinal format (1 is 1-1-1, 2 is 1-1-2 and so on).

    WARNING: Only works on single elements, not on entire columns, so use the `.apply()` method.
    """
    return datetime.date.fromisoformat(
        datetime.datetime.strptime(in_date, "%m/%d/%Y").strftime("%Y-%m-%d")
    ).toordinal()


def combinedateandtime(in_date: str, in_time: str) -> str:
    return f"{in_date} {in_time}:00"
