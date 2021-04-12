from scenarios.common import inside_rectangle_formula, outside_rectangle_formula
import numpy as np
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
                    not_at_obstacle.always(0,T) & \
                    velocity_bounded.always(0,T) & \
                    in_workspace.always(0,T) & \
                    at_goal.eventually(0,T)

    return specification

def plot_reach_avoid_scenario(goal_bounds, obstacle_bounds):
    # Get locations, heights, and widths
    obs_x = obstacle_bounds[0]
    obs_y = obstacle_bounds[2]
    obs_w = obstacle_bounds[1]-obs_x
    obs_h = obstacle_bounds[3]-obs_y
    
    goal_x = goal_bounds[0]
    goal_y = goal_bounds[2]
    goal_w = goal_bounds[1]-goal_x
    goal_h = goal_bounds[3]-goal_y

    # Make rectangular patches
    obstacle = Rectangle((obs_x,obs_y),obs_w,obs_h, color='red', alpha=0.5)
    goal = Rectangle((goal_x,goal_y),goal_w,goal_h, color='green', alpha=0.5)

    ax = plt.gca()
    ax.add_patch(obstacle)
    ax.add_patch(goal)

    # set the field of view
    ax.set_xlim((0,12))
    ax.set_ylim((0,12))
    ax.set_aspect('equal')

if __name__=="__main__":

    # Define the regions of interest
    goal_bounds = (7,8,8,9)     # (xmin, xmax, ymin, ymax)
    obstacle_bounds = (3,5,4,6)
    
    # Test creation of a specification
    spec = reach_avoid_specification(goal_bounds, obstacle_bounds, T=10)

    # Plot the scenario at hand
    plot_reach_avoid_scenario(goal_bounds, obstacle_bounds)
    plt.show()
