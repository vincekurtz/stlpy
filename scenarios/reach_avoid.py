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
    # Control bounds
    u_min = -0.5
    u_max = 0.5
    control_bounded = inside_rectangle_formula((u_min,u_max,u_min,u_max), 4, 5, 6)

    # Goal Reaching
    at_goal = inside_rectangle_formula(goal_bounds, 0, 1, 6)

    # Obstacle Avoidance
    not_at_obstacle = outside_rectangle_formula(obstacle_bounds, 0, 1, 6)

    # Velocity bounded
    v_min = -1.0
    v_max = 1.0
    velocity_bounded = inside_rectangle_formula((v_min, v_max, v_min, v_max), 2, 3, 6)

    # Workspace boundaries
    x_min = 0; x_max = 10;
    y_min = 0; y_max = 10;
    in_workspace = inside_rectangle_formula((x_min, x_max, y_min, y_max), 0, 1, 6)

    # Put all of the constraints together in one specification
    specification = control_bounded.always(0,T) & \
                    velocity_bounded.always(0,T) & \
                    in_workspace.always(0,T) & \
                    not_at_obstacle.until(at_goal, 0, T)

    return specification

def plot_reach_avoid_scenario(goal_bounds, obstacle_bounds):
    ax = plt.gca()

    # Make and add rectangular patches
    obstacle = make_rectangle_patch(*obstacle_bounds, color='red', alpha=0.5)
    goal = make_rectangle_patch(*goal_bounds, color='green', alpha=0.5)
    ax.add_patch(obstacle)
    ax.add_patch(goal)

    # set the field of view
    ax.set_xlim((0,12))
    ax.set_ylim((0,12))
    ax.set_aspect('equal')
