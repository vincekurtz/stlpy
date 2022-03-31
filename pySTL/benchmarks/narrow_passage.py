from .common import (inside_rectangle_formula,
                     outside_rectangle_formula,
                     make_rectangle_patch)
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

##
#
# Tools for setting up a slightly more complex specification where a robot
# with double integrator dynamics must navigate through several obstacles
# with narrow passages before reaching one of several possible goals.
#
##

def narrow_passage_specification(T):
    """
    Return an STLFormula that describes this scenario, with time bound T.

    We'll assume that the robot has double integrator dynamics, i.e.,

        x = [px,py,pdx,pdy], u = [pddx, pddy]

    and that the output signal is given by y = [x;u].
    """
    obstacles = [(2,5,4,6),
                 (5.5,9,3.8,5.7),
                 (4.6,8,0.5,3.5),
                 (2.2,4.4,6.4,11)]
    goals = [(7,8,8,9),
             (9.5,10.5,1.5,2.5)]

    # Goal Reaching
    goal_formulas = []
    for goal in goals:
        goal_formulas.append(inside_rectangle_formula(goal, 0, 1, 6))

    at_any_goal = goal_formulas[0]
    for i in range(1,len(goal_formulas)):
        at_any_goal = at_any_goal | goal_formulas[i]

    # Obstacle Avoidance
    obstacle_formulas = []
    for obs in obstacles:
        obstacle_formulas.append(outside_rectangle_formula(obs, 0, 1, 6))

    obstacle_avoidance = obstacle_formulas[0]
    for i in range(1, len(obstacle_formulas)):
        obstacle_avoidance = obstacle_avoidance & obstacle_formulas[i]

    # Put all of the constraints together in one specification
    specification = at_any_goal.eventually(0,T) & \
                    obstacle_avoidance.always(0,T)

    return specification

def plot_narrow_passage_scenario():
    ax = plt.gca()

    obstacles = [(2,5,4,6),
                 (5.5,9,3.8,5.7),
                 (4.6,8,0.5,3.5),
                 (2.2,4.4,6.4,11)]
    goals = [(7,8,8,9),
             (9.5,10.5,1.5,2.5)]

    # Make and add rectangular patches
    for obstacle in obstacles:
        ax.add_patch(make_rectangle_patch(*obstacle, color='k', alpha=0.5))
    for goal in goals:
        ax.add_patch(make_rectangle_patch(*goal, color='green', alpha=0.5))

    # set the field of view
    ax.set_xlim((0,12))
    ax.set_ylim((0,12))
    ax.set_aspect('equal')
