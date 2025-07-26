# src/utils.py
import argparse


def min_100(value):
    """
    Validates that the input value is an integer greater
    than or equal to 100.

    This function attempts to convert the given value to
    an integer and checks whether it is at least 100. If
    the value is less than 100, it raises an `argparse.ArgumentTypeError`
    to signal invalid input for command-line arguments.

    Args:
        value (str or int): The input value to validate, typically
        provided as astring from command-line arguments.

    Returns:
        int: The validated integer value if it is greater than
             or equal to 100.

    Raises:
        argparse.ArgumentTypeError: If the input value is
        less than 100.

    Example:
        >>> min_100("150")
        150
        >>> min_100(200)
        200
        >>> min_100("50")
        Traceback (most recent call last):
            ...
        argparse.ArgumentTypeError: Minimum allowed episodes
        is 100 (you entered 50).
    """
    ivalue = int(value)
    if ivalue < 100:
        raise argparse.ArgumentTypeError(f"Minimum allowed episodes "
                                         f"is 100 (you entered {ivalue}).")
    return ivalue
