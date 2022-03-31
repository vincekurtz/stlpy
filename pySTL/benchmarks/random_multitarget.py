from .common import (inside_rectangle_formula,
                     outside_rectangle_formula,
                     make_rectangle_patch)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

##
#
# Tools for setting up a specification where a robot with double 
# integrator dynamics must navigate through a field of obstacles
# and reach targets from multiple sets. 
#
##

def random_multitarget_specification(num_obstacles, num_groups, targets_per_group, T, seed=None):
    """
    Return an STLFormula that describes this scenario, with time bound T.
    
    We'll assume that the robot has double integrator dynamics, i.e.,
    
        x = [px,py,pdx,pdy], u = [pddx, pddy]
    
    and that the output signal is given by y = [x;u].

    In addition to the specification, this function returns a list
    of obstacles and a list of target groups, where each element
    of each target group is itself a target, and at least one target
    per group must be visited over the course of T timesteps. 
    """
    # Set the seed for the random number generator (for reproducability)
    np.random.seed(seed)

    # Create the (randomly generated) set of obstacles
    obstacles = []
    for i in range(num_obstacles):
        x = np.random.uniform(0,9)  # keep within workspace
        y = np.random.uniform(0,9)
        obstacles.append((x,x+2,y,y+2))
   
    # Create the (randomly generated) set of targets
    targets = []
    for i in range(num_groups):
        target_group = []
        for j in range(targets_per_group):
            x = np.random.uniform(0,9)
            y = np.random.uniform(0,9)
            target_group.append((x,x+1,y,y+1))
        targets.append(target_group)

    # Specify that we must avoid all obstacles
    obstacle_formulas = []
    for obs in obstacles:
        obstacle_formulas.append(outside_rectangle_formula(obs, 0, 1, 6))
    obstacle_avoidance = obstacle_formulas[0]
    for i in range(1, len(obstacle_formulas)):
        obstacle_avoidance = obstacle_avoidance & obstacle_formulas[i]

    # Specify that for each target group, we need to visit at least one
    # of the targets in that group
    target_group_formulas = []
    for target_group in targets:
        group_formulas = []
        for target in target_group:
            group_formulas.append(inside_rectangle_formula(target, 0, 1, 6))
        reach_target_group = group_formulas[0]
        for i in range(1, targets_per_group):
            reach_target_group = reach_target_group | group_formulas[i]
        target_group_formulas.append(reach_target_group)

    # Put all of the constraints together in one specification
    specification = obstacle_avoidance.always(0,T)
    for reach_target_group in target_group_formulas:
        specification = specification & reach_target_group.eventually(0,T)
    
    return specification, obstacles, targets

def plot_random_multitarget_scenario(obstacles, targets):
    ax = plt.gca()

    # Add red rectangles for the obstacles
    for obstacle in obstacles:
        ax.add_patch(make_rectangle_patch(*obstacle, color='k', alpha=0.5, zorder=-1))

    # Use the color cycle to choose the colors of each target group
    # (note that this won't work for more than 10 target groups)
    colors = plt.cm.tab10.colors
    for i, target_group in enumerate(targets):
        color = colors[i]
        for target in target_group:
            ax.add_patch(make_rectangle_patch(*target, color=color, alpha=0.7, zorder=-1))

    # set the field of view
    ax.set_xlim((0,10))
    ax.set_ylim((0,10))
    ax.set_aspect('equal')
