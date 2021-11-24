from scenarios.common import *
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
    # Control bounds
    u_min = -0.5
    u_max = 0.5
    control_bounded = inside_rectangle_formula((u_min,u_max,u_min,u_max), 4, 5, 6)

    # Goal Reaching
    at_goal = inside_rectangle_formula(goal, 0, 1, 6)

    # Target reaching
    at_target_one = inside_rectangle_formula(target_one, 0, 1, 6)
    at_target_two = inside_rectangle_formula(target_two, 0, 1, 6)
    at_either_target = at_target_one | at_target_two

    # Obstacle Avoidance
    not_at_obstacle = outside_rectangle_formula(obstacle, 0, 1, 6)

    # Velocity bounded
    v_min = -1.0
    v_max = 1.0
    velocity_bounded = inside_rectangle_formula((v_min, v_max, v_min, v_max), 2, 3, 6)

    # Workspace boundaries
    x_min = 0; x_max = 10;
    y_min = 0; y_max = 10;
    in_workspace = inside_rectangle_formula((x_min, x_max, y_min, y_max), 0, 1, 6)

    specification = control_bounded.always(0,T) & \
                    velocity_bounded.always(0,T) & \
                    in_workspace.always(0,T) & \
                    at_either_target.eventually(0,T) & \
                    not_at_obstacle.until(at_goal, 0, T)

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
