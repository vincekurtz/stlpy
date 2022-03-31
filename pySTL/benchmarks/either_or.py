from .common import (inside_rectangle_formula,
                     outside_rectangle_formula,
                     make_rectangle_patch)
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def either_or_specification(goal, target_one, target_two, obstacle, T):
    """
    Return an STLFormula that describes the "either-or" scenario, where
    a robot with double integrator dynamics must reach one of two targets
    and avoid an obstacle before reaching the goal. 

    All regions are assumed to be rectangles specified by a tuple

        (xmin, xmax, ymin, ymax)

    """
    # Goal Reaching
    at_goal = inside_rectangle_formula(goal, 0, 1, 6)

    # Target reaching
    at_target_one = inside_rectangle_formula(target_one, 0, 1, 6).always(0,5)
    at_target_two = inside_rectangle_formula(target_two, 0, 1, 6).always(0,5)
    at_either_target = at_target_one | at_target_two

    # Obstacle Avoidance
    not_at_obstacle = outside_rectangle_formula(obstacle, 0, 1, 6)

    specification = at_either_target.eventually(0,T-5) & \
                    not_at_obstacle.always(0,T) & \
                    at_goal.eventually(0,T)

    return specification

def plot_either_or_scenario(goal, target_one, target_two, obstacle):
    ax = plt.gca()

    # Make and add rectangular patches
    ax.add_patch(make_rectangle_patch(*obstacle, color='k', alpha=0.5))
    ax.add_patch(make_rectangle_patch(*target_one, color='blue', alpha=0.5))
    ax.add_patch(make_rectangle_patch(*target_two, color='blue', alpha=0.5))
    ax.add_patch(make_rectangle_patch(*goal, color='green', alpha=0.5))

    # set the field of view
    ax.set_xlim((0,10))
    ax.set_ylim((0,10))
    ax.set_aspect('equal')
