##
#
# Common helper functions for defining STL specifications.
#
##

import numpy as np
from stlpy.STL import LinearPredicate, NonlinearPredicate
from matplotlib.patches import Rectangle, Circle

def inside_circle_formula(center, radius, y1_index, y2_index, d, name=None):
    """
    Create an STL formula representing being inside a
    circle with the given center and radius.

    :param center:      Tuple ``(y1, y2)`` specifying the center of the
                        circle.
    :param radius:      Radius of the circle
    :param y1_index:    index of the first (``y1``) dimension
    :param y2_index:    index of the second (``y2``) dimension
    :param d:           dimension of the overall signal
    :param name:        (optional) string describing this formula

    :return inside_circle:   A ``NonlinearPredicate`` specifying being inside the
                             circle at time zero.
    """
    # Define the predicate function g(y) >= 0
    def g(y):
        y1 = y[y1_index]
        y2 = y[y2_index]
        return radius**2 - (y1-center[0])**2 - (y2-center[1])**2

    return NonlinearPredicate(g, d, name=name)


def inside_rectangle_formula(bounds, y1_index, y2_index, d, name=None):
    """
    Create an STL formula representing being inside a
    rectangle with the given bounds:

    ::

       y2_max   +-------------------+
                |                   |
                |                   |
                |                   |
       y2_min   +-------------------+
                y1_min              y1_max

    :param bounds:      Tuple ``(y1_min, y1_max, y2_min, y2_max)`` containing
                        the bounds of the rectangle. 
    :param y1_index:    index of the first (``y1``) dimension
    :param y2_index:    index of the second (``y2``) dimension
    :param d:           dimension of the overall signal
    :param name:        (optional) string describing this formula

    :return inside_rectangle:   An ``STLFormula`` specifying being inside the
                                rectangle at time zero.
    """
    assert y1_index < d , "index must be less than signal dimension"
    assert y2_index < d , "index must be less than signal dimension"

    # Unpack the bounds
    y1_min, y1_max, y2_min, y2_max = bounds

    # Create predicates a*y >= b for each side of the rectangle
    a1 = np.zeros((1,d)); a1[:,y1_index] = 1
    right = LinearPredicate(a1, y1_min)
    left = LinearPredicate(-a1, -y1_max)

    a2 = np.zeros((1,d)); a2[:,y2_index] = 1
    top = LinearPredicate(a2, y2_min)
    bottom = LinearPredicate(-a2, -y2_max)

    # Take the conjuction across all the sides
    inside_rectangle = right & left & top & bottom

    # set the names
    if name is not None:
        right.name = "right of " + name
        left.name = "left of " + name
        top.name = "top of " + name
        bottom.name = "bottom of " + name
        inside_rectangle.name = name

    return inside_rectangle

def outside_rectangle_formula(bounds, y1_index, y2_index, d, name=None):
    """
    Create an STL formula representing being outside a
    rectangle with the given bounds:

    ::

       y2_max   +-------------------+
                |                   |
                |                   |
                |                   |
       y2_min   +-------------------+
                y1_min              y1_max

    :param bounds:      Tuple ``(y1_min, y1_max, y2_min, y2_max)`` containing
                        the bounds of the rectangle. 
    :param y1_index:    index of the first (``y1``) dimension
    :param y2_index:    index of the second (``y2``) dimension
    :param d:           dimension of the overall signal
    :param name:        (optional) string describing this formula
    
    :return outside_rectangle:   An ``STLFormula`` specifying being outside the
                                 rectangle at time zero.
    """
    assert y1_index < d , "index must be less than signal dimension"
    assert y2_index < d , "index must be less than signal dimension"

    # Unpack the bounds
    y1_min, y1_max, y2_min, y2_max = bounds

    # Create predicates a*y >= b for each side of the rectangle
    a1 = np.zeros((1,d)); a1[:,y1_index] = 1
    right = LinearPredicate(a1, y1_max)
    left = LinearPredicate(-a1, -y1_min)

    a2 = np.zeros((1,d)); a2[:,y2_index] = 1
    top = LinearPredicate(a2, y2_max)
    bottom = LinearPredicate(-a2, -y2_min)

    # Take the disjuction across all the sides
    outside_rectangle = right | left | top | bottom

    # set the names
    if name is not None:
        right.name = "right of " + name
        left.name = "left of " + name
        top.name = "top of " + name
        bottom.name = "bottom of " + name
        outside_rectangle.name = name

    return outside_rectangle

def make_rectangle_patch(xmin, xmax, ymin, ymax, **kwargs):
    """
    Convienience function for making a ``matplotlib.patches.Rectangle`` 
    patch for visualizing a rectangle:

    ::

       ymax   +-------------------+
              |                   |
              |                   |
              |                   |
       ymin   +-------------------+
              xmin                xmax

    :param xmin:        horizontal lower bound of the rectangle.
    :param xmax:        horizontal upper bound of the rectangle.
    :param ymin:        vertical lower bound of the rectangle.
    :param ymax:        vertical upper bound of the rectangle.
    :param kwargs:    (optional) keyword arguments passed to
                        the ``Rectangle`` constructor.

    :return patch:  a ``matplotlib.patches.Rectangle`` patch.

    """
    x = xmin
    y = ymin
    width = xmax-x
    height = ymax-y

    return Rectangle((x,y), width, height, **kwargs)

def make_circle_patch(center, radius, **kwargs):
    """
    Convienience function for making a ``matplotlib.patches.Circle`` 
    patch for visualizing a circle with the given center and radius.

    :param center:  Tuple containing the center coordinates of the circle
    :param radius:  The circle's radius
    :param kwargs:  (optional) keyword arguments passed to
                        the ``Circle`` constructor.

    :return patch:  a ``matplotlib.patches.Circle`` patch.
    """
    return Circle(center, radius, **kwargs)
