##
#
# Common helper functions for defining STL specifications.
#
##

import numpy as np
from STL import STLFormula, STLPredicate

def inside_rectangle_formula(bounds, y1_index, y2_index, d):
    """
    Create an STL formula representing being inside the
    rectangle given by the bounds

        (y1_min, y1_max, y2_min, y2_max),

    on the where y1 and y2 are elements of the d-dimensional
    signal y, and their indices are specified. 
    """
    assert y1_index < d , "index must be less than signal dimension"
    assert y2_index < d , "index must be less than signal dimension"

    # Unpack the bounds
    y1_min, y1_max, y2_min, y2_max = bounds

    # Create predicates a*y >= b for each side of the rectangle
    a1 = np.zeros((1,d)); a1[:,y1_index] = 1
    right = STLPredicate(a1, y1_min)
    left = STLPredicate(-a1, -y1_max)

    a2 = np.zeros((1,d)); a2[:,y2_index] = 1
    top = STLPredicate(a2, y2_min)
    bottom = STLPredicate(-a2, -y2_max)

    # Take the conjuction across all the sides
    inside_rectangle = right & left & top & bottom
    return inside_rectangle

def outside_rectangle_formula(bounds, y1_index, y2_index, d):
    """
    Create an STL formula representing being outside the
    rectangle given by the bounds

        (y1_min, y1_max, y2_min, y2_max),

    on the where y1 and y2 are elements of the d-dimensional
    signal y, and their indices are specified. 
    """
    assert y1_index < d , "index must be less than signal dimension"
    assert y2_index < d , "index must be less than signal dimension"

    # Unpack the bounds
    y1_min, y1_max, y2_min, y2_max = bounds

    # Create predicates a*y >= b for each side of the rectangle
    a1 = np.zeros((1,d)); a1[:,y1_index] = 1
    right = STLPredicate(a1, y1_max)
    left = STLPredicate(-a1, -y1_min)

    a2 = np.zeros((1,d)); a2[:,y2_index] = 1
    top = STLPredicate(a2, y2_max)
    bottom = STLPredicate(-a2, -y2_min)

    # Take the disjuction across all the sides
    outside_rectangle = right | left | top | bottom
    return outside_rectangle
