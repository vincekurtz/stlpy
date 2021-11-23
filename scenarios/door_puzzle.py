from scenarios.common import *
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

##
#
# Tools for setting up a door-puzzle specification, inspired by
#
#   https://arxiv.org/pdf/1806.00805.pdf
#   https://daweisun.me/static/MA-STL.pdf
#
# where a robot must pick up a set of keys from different
# locations before passing through a set of doors to reach a goal.
#
##

def door_puzzle_specification(T, N_pairs):
    """
    Return an STLFormula that describes this scenario. We'll assume that 
    the robot has double integrator dynamics, i.e.,
    
        x = [px,py,pdx,pdy], u = [pddx, pddy]
    
    and that the output signal is given by y = [x;u].
    """
    assert N_pairs >= 1 and N_pairs <= 4, "number of pairs must be between 1 and 4"
    goal_bounds = (14.1,14.9,4.1,5.9)

    obs1_bounds = (8,15.01,-0.01,4)
    obs2_bounds = (8,15.01,6,10.01)

    door1_bounds = (12.8,14,3.99,6.01)
    door2_bounds = (11.5,12.7,3.99,6.01)
    door3_bounds = (10.2,11.4,3.99,6.01)
    door4_bounds = (8.9,10.1,3.99,6.01)

    key1_bounds = (1,2,1,2)
    key2_bounds = (1,2,8,9)
    key3_bounds = (6,7,8,9)
    key4_bounds = (6,7,1,2)

    # Control bounds
    u_min = -0.5
    u_max = 0.5
    control_bounded = inside_rectangle_formula((u_min,u_max,u_min,u_max), 4, 5, 6)
    
    # Velocity bounded
    v_min = -1.0
    v_max = 1.0
    velocity_bounded = inside_rectangle_formula((v_min, v_max, v_min, v_max), 2, 3, 6)

    # Workspace boundaries
    x_min = 0; x_max = 15;
    y_min = 0; y_max = 10;
    in_workspace = inside_rectangle_formula((x_min, x_max, y_min, y_max), 0, 1, 6)

    # Goal Reaching
    at_goal = inside_rectangle_formula(goal_bounds, 0, 1, 6)

    # Obstacle Avoidance
    not_at_obs1 = outside_rectangle_formula(obs1_bounds, 0, 1, 6)
    not_at_obs2 = outside_rectangle_formula(obs2_bounds, 0, 1, 6)
    obstacle_avoidance = not_at_obs1 & not_at_obs2

    # Key-door constraints
    no_door1 = outside_rectangle_formula(door1_bounds, 0, 1, 6)
    key1 = inside_rectangle_formula(key1_bounds, 0, 1, 6)
    k1d1 = no_door1.until(key1, 0, T)
    
    no_door2 = outside_rectangle_formula(door2_bounds, 0, 1, 6)
    key2 = inside_rectangle_formula(key2_bounds, 0, 1, 6)
    k2d2 = no_door2.until(key2, 0, T)
    
    no_door3 = outside_rectangle_formula(door3_bounds, 0, 1, 6)
    key3 = inside_rectangle_formula(key3_bounds, 0, 1, 6)
    k3d3 = no_door3.until(key3, 0, T)
    
    no_door4 = outside_rectangle_formula(door4_bounds, 0, 1, 6)
    key4 = inside_rectangle_formula(key4_bounds, 0, 1, 6)
    k4d4 = no_door4.until(key4, 0, T)

    if N_pairs == 1:
        key_constraints = k1d1
    elif N_pairs == 2:
        key_constraints = k1d1 & k2d2
    elif N_pairs == 3:
        key_constraints = k1d1 & k2d2 & k3d3
    elif N_pairs == 4:
        key_constraints = k1d1 & k2d2 & k3d3 & k4d4
    else:
        raise ValueError("Invalid number of key-door pairs: %s" % N_pairs)

    # Put all of the constraints together in one specification
    specification = control_bounded.always(0,T) & \
                    velocity_bounded.always(0,T) & \
                    in_workspace.always(0,T) & \
                    obstacle_avoidance.always(0,T) & \
                    key_constraints & \
                    at_goal.eventually(0,T)

    return specification

def plot_door_puzzle_scenario(N_pairs):
    goal_bounds = (14.1,14.9,4.1,5.9)

    obs1_bounds = (8,15.01,-0.01,4)
    obs2_bounds = (8,15.01,6,10.01)

    door1_bounds = (12.8,14,3.99,6.01)
    door2_bounds = (11.5,12.7,3.99,6.01)
    door3_bounds = (10.2,11.4,3.99,6.01)
    door4_bounds = (8.9,10.1,3.99,6.01)

    key1_bounds = (1,2,1,2)
    key2_bounds = (1,2,8,9)
    key3_bounds = (6,7,8,9)
    key4_bounds = (6,7,1,2)

    ax = plt.gca()

    # Make and add rectangular patches
    obs1 = make_rectangle_patch(*obs1_bounds, color='k', alpha=0.5)
    obs2 = make_rectangle_patch(*obs2_bounds, color='k', alpha=0.5)
    door1 = make_rectangle_patch(*door1_bounds, color='red', alpha=0.5)
    door2 = make_rectangle_patch(*door2_bounds, color='red', alpha=0.5)
    door3 = make_rectangle_patch(*door3_bounds, color='red', alpha=0.5)
    door4 = make_rectangle_patch(*door4_bounds, color='red', alpha=0.5)
    key1 = make_rectangle_patch(*key1_bounds, color='blue', alpha=0.5)
    key2 = make_rectangle_patch(*key2_bounds, color='blue', alpha=0.5)
    key3 = make_rectangle_patch(*key3_bounds, color='blue', alpha=0.5)
    key4 = make_rectangle_patch(*key4_bounds, color='blue', alpha=0.5)
    goal = make_rectangle_patch(*goal_bounds, color='green', alpha=0.5)

    ax.add_patch(obs1)
    ax.add_patch(obs2)
    ax.add_patch(goal)

    if N_pairs >= 1:
        ax.add_patch(door1)
        ax.add_patch(key1)

    if N_pairs >= 2:
        ax.add_patch(door2)
        ax.add_patch(key2)

    if N_pairs >= 3:
        ax.add_patch(door3)
        ax.add_patch(key3)

    if N_pairs >= 4:
        ax.add_patch(door4)
        ax.add_patch(key4)

    # set the field of view
    ax.set_xlim((0,15))
    ax.set_ylim((0,10))
    ax.set_aspect('equal')
