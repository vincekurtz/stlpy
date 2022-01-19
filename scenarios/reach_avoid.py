from scenarios.common import *
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

##
#
# Tools for setting up a simple reach-avoid specification, 
# i.e., reach a rectangular goal region and avoid a rectangular obstacle. 
#
##

def reach_avoid_specification(goal_bounds, obstacle_bounds, T):
    """
    Return an STLFormula that describes this scenario, where goal_bounds
    and obstacle_bounds are tuples containing (xmin, xmax, ymin, ymax)
    for the rectangular regions of interest. 
    
    We'll assume that the robot has double integrator dynamics, i.e.,
    
        x = [px,py,pdx,pdy], u = [pddx, pddy]
    
    and that the output signal is given by y = [x;u].
    """
    # Goal Reaching
    at_goal = inside_rectangle_formula(goal_bounds, 0, 1, 6)

    # Obstacle Avoidance
    not_at_obstacle = outside_rectangle_formula(obstacle_bounds, 0, 1, 6)

    # Put all of the constraints together in one specification
    #specification = not_at_obstacle.until(at_goal, 0, T)
    specification = not_at_obstacle.always(0,T) & at_goal.eventually(T,T)

    return specification

def plot_reach_avoid_scenario(goal_bounds, obstacle_bounds):
    ax = plt.gca()

    # Make and add rectangular patches
    obstacle = make_rectangle_patch(*obstacle_bounds, color='k', alpha=0.5)
    goal = make_rectangle_patch(*goal_bounds, color='green', alpha=0.5)
    ax.add_patch(obstacle)
    ax.add_patch(goal)

    # set the field of view
    ax.set_xlim((0,10))
    ax.set_ylim((0,10))
    ax.set_aspect('equal')
