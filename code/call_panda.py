import os
import subprocess

import numpy as np
from dotenv import load_dotenv
from loguru import logger

load_dotenv("path.env")
PANDA = os.getenv("PANDA")  # path to the compiled PANDA executable

PANDA_DIR = "data/panda/"


def float_to_rational(
    value: float, precision: float = 1e-12, max_integer_bit_size: int = 64, iterations: int = 100
) -> tuple[int, int]:
    """
    Convert a floating-point number to a rational representation.

    :param value: The float value to convert.
    :param precision: The absolute precision to which the float is being cast.
    :param max_integer_bit_size: The maximum bit size for integer numerators/denominators.
    :return: A tuple (numerator, denominator) representing the rational.
    :raises OverflowError: If the rational representation exceeds the maximum bit size.
    """
    # AI GENERATED CODE

    if abs(value) < precision:
        return 0, 1

    # Use continued fractions to find the best rational approximation
    # This is a simplified version and may not cover all edge cases
    sign = -1 if value < 0 else 1
    value = abs(value)

    # Find the integer part
    integer_part = int(value)
    fractional_part = value - integer_part

    # Use a simple method to find a good enough rational approximation
    numerator = integer_part
    denominator = 1

    for _ in range(iterations):  # Limit iterations to avoid infinite loops
        if fractional_part < precision:
            break

        # Update the denominator
        denominator *= 10
        fractional_part *= 10
        digit = int(fractional_part)
        numerator = numerator * 10 + digit
        fractional_part -= digit

    # Check for overflow
    if abs(numerator) >= (1 << max_integer_bit_size) or abs(denominator) >= (
        1 << max_integer_bit_size
    ):
        raise OverflowError("Rational representation exceeds maximum bit size.")

    if (value - (sign * numerator / denominator)) > precision:
        raise OverflowError(
            f"Value {value} couldn't be represented as a rational within the given precision after {iterations} iterations."  # noqa: E501
        )

    logger.debug(f"Converted {value} to rational {sign * numerator}/{denominator}")
    return sign * numerator, denominator


def cast_array_to_rationals(
    arr: np.ndarray,
    precision: float = 1e-12,
    max_integer_bit_size: int = 64,
) -> list[list[str]]:
    """
    Given a 2D array containing rows of floats, cast each element to a literal rational expression
    within floating-point precision.

    :param arr: The input 2D numpy array.
    :param precision: The absolute precision to which floats are being cast to rationals.
    :param max_integer_bit_size: The maximum bit size for integer numerators/denominators.
    :return: A list of lists of strings, containing for each row its elements as stringified rational expressions.
    :raises ValueError: If the input array is not a 2D array of floats
    :raises OverflowError: If a float cannot be reasonably cast to a rational (numerator/denominator too large)
    """  # noqa: E501
    if (
        not isinstance(arr, np.ndarray)
        or arr.ndim != 2
        or not np.issubdtype(arr.dtype, np.floating)
    ):
        raise ValueError("Input must be a 2D numpy array of floats.")

    result = []
    for row in arr:
        row_result = []
        for value in row:
            if abs(value) < precision:
                row_result.append("0")
            else:
                # Attempt to cast the float to a rational
                try:
                    numerator, denominator = float_to_rational(
                        value, precision, max_integer_bit_size
                    )
                    row_result.append(f"{numerator}/{denominator}")
                except OverflowError:
                    raise OverflowError(f"Value {value} cannot be cast to a rational.")
        result.append(row_result)
    return result


def format_input(input_file: str) -> tuple[str, str]:
    """
    Format the vertices we usually produce to a properly formatted input file for PANDA.
    Returns the path to the formatted input file, and a suggested path to an output file.

    :param input_file: The path to the input file containing the vertices. Assumed to be a `.txt` file.
    :return: The path to the formatted input file, and a suggested path to an output file.
    """  # noqa: E501
    # 1. Try to name files
    formatted_input_file = f"{PANDA_DIR}in/{input_file.split('.')[0]}"
    suggested_output_file = f"{PANDA_DIR}out/{input_file.split('.')[0]}"

    count = 0
    while os.path.exists(formatted_input_file) or os.path.exists(suggested_output_file):
        count += 1
        if formatted_input_file.endswith(str(count - 1)):
            formatted_input_file = formatted_input_file[:-1] + str(count)
            suggested_output_file = suggested_output_file[:-1] + str(count)
        else:
            formatted_input_file += str(count)
            suggested_output_file += str(count)

    # 2. Load data
    loading_seen = set()
    loaded_points = []

    with open(input_file, "r") as f:
        # Load the vertices from the file
        while True:
            line = f.readline()
            if not line:
                break

            # Convert the line to a numpy array and deduplicate
            if ";" in line:
                # If the line contains info, split it
                _, point_str = line.split(";")
                point = np.array([float(x) for x in point_str.strip().split(",")])
            else:
                # If the line does not contain info, just parse the point
                point = np.array([float(x) for x in line.strip().split(",")])
            if point.tobytes() not in loading_seen:
                loading_seen.add(point.tobytes())
                loaded_points.append(point)

    points_array = np.array(loaded_points)
    rationals = cast_array_to_rationals(points_array)

    with open(formatted_input_file, "w") as f:
        f.write("Vertices:\n")
        for row in rationals:
            f.write(" ".join(row) + "\n")

    # Implementation goes here
    return formatted_input_file, suggested_output_file


def call_panda(input_file: str, output_file: str) -> None:
    """
    Call panda on the formatted inputs to perform computations
    and generate outputs in the output file.

    :param input_file: The path to the formatted input file.
    :param output_file: The path to the output file where results will be written.
    :return: `None`
    :raise RuntimeError: If the PANDA environment variable is not set or could not be loaded.
    """
    if PANDA is None:
        raise RuntimeError("PANDA environment variable is not set or could not be loaded.")
    subprocess.run([PANDA, input_file, output_file], check=True)


def run_panda(input_file: str) -> str:
    """
    Run the PANDA tool on the given input file and return the path to the output file.
    """
    _in, _out = format_input(input_file)
    call_panda(_in, _out)
    return _out


if __name__ == "__main__":
    print(f"PANDA executable path: {PANDA}")

    out = run_panda("full_222.txt")
    logger.success(f"PANDA run completed successfully. Output file: {out}")
