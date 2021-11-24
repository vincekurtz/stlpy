from scenarios.common import *
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

    # Control bounds
    u_min = -0.5
    u_max = 0.5
    control_bounded = inside_rectangle_formula((u_min,u_max,u_min,u_max), 4, 5, 6)

    # Velocity bounds
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
                    at_any_goal.eventually(0,T) & \
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
