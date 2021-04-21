import numpy as np


def show_utility(values: np.ndarray):
    """Prints the utility array to the screen."""
    row_divider = "-" * ((8 * values.shape[0]) + values.shape[0] + 1)
    for row in range(values.shape[0]):
        print(row_divider)
        out = "| "
        for col in range(self.utility.shape[1]):
            out += str(round(self.utility[(row, col)], 2)).ljust(6) + ' | '
        print(out)
    print(row_divider)


def show_policy(utility: np.ndarray):
    """"""
    ...