#
# Python Utility Functions
#

import time

# Author: Bex Tuychiev
def timer(func):
    """
    A decorator to calculate how long a function runs.

    Parameters
    ----------
    func: callable
      The function being decorated.

    Returns
    -------
    func: callable
      The decorated function.
    """

    def wrapper(*args, **kwargs):
        # Start the timer
        start = time.time()
        # Call the `func`
        result = func(*args, **kwargs)
        # End the timer
        end = time.time()

        print(f"{func.__name__} took {round(end - start, 4)} "
              "seconds to run!")
        return result

    return wrapper