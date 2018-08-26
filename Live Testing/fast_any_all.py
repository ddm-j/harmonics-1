import numpy as np
import functools

# Purpose:
# np.any and np.all are quite slow - about 15x slower than using logical_or and logical_and calls
# This code tries to combine the best features of np.logical_and/or and np.any/all.
# It takes lists of arrays (like np.any/all), but is fast like np.logical_or/and
# It doesn't have the flexibility (in terms of axis choice) that is offered by np.any/all.

"""This module offers a very fast alternative to numpy.any/all, for the use case of a list of bool ndarrays"""

__version__ = "1.0.0"


def any(inputs):
    """faa.any([list of boolean ndarrays]), returns true where at least one element is true in an ndarray at that position.
    """
    return fast_logic(inputs, np.logical_or)


def all(inputs):
    """faa.all([list of boolean ndarrays]), returns true where at least one element is true in an ndarray at that position.
    """
    return fast_logic(inputs, np.logical_and)


def fast_logic(inputs, logic_func):
    """ Generic helper function for any() and all()
        No direct type-checking; relies on numpy.boolean_or/and.
    """

    # Catch empty input, retain compatibility with np.any return value.
    if len(inputs) == 0:
        return False

    output = inputs[0]  # separate out first element

    for w in range(1, len(inputs)):  # logical combination with other elements
        output = logic_func(output, inputs[w])  # note that this loop doesn't run when len(inputs)=1.

    return output


def reduce_any(inputs):
    return functools.reduce(np.logical_or, inputs, False)


def boolean_index_any(inputs):
    # Catch empty input, retain compatibility with np.any return value.
    if len(inputs) == 0:
        return False

    output = np.copy(inputs[0])  # clone first element

    # overlay layers as masks using boolean indexing

    for w in range(1, len(inputs)):
        output[inputs[w]] = True

    return output
