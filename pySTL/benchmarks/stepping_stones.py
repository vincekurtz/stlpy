from .common import (inside_rectangle_formula,
                     outside_rectangle_formula,
                     make_rectangle_patch)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

##
#
# Tools for setting up a specification where a robot with double 
# integrator dynamics must navigate to a goal while only stepping
# on certain predefined spaces. 
#
##

def stepping_stones_specification(num_stones, T, seed=None):
    """
    Return an STLFormula that describes this scenario, with time bound T.
    
    We'll assume that the robot has double integrator dynamics, i.e.,
    
        x = [px,py,pdx,pdy], u = [pddx, pddy]
    
    and that the output signal is given by y = [x;u].

    In addition to the specification, this function returns a list
    of stepping stones, where the last stone in this list is the target.
    """
    # Set the seed for the random number generator (for reproducability)
    np.random.seed(seed)

    # Create the (randomly generated) set of stepping stones
    stones = []
    for i in range(num_stones):
        x = np.random.uniform(0,9)  # keep within workspace
        y = np.random.uniform(0,9)
        stones.append((x,x+1,y,y+1))
   
    # Specify the target/goal
    target = stones[-1]

    # Specify that we must be on any one of the stones
    stone_formulas = []
    for stone in stones:
        stone_formulas.append(inside_rectangle_formula(stone, 0, 1, 6))

    on_any_stone = stone_formulas[0]
    for i in range(1, len(stone_formulas)):
        on_any_stone = on_any_stone | stone_formulas[i]

    # Specify that we much reach the target
    reach_target = inside_rectangle_formula(target, 0, 1, 6)

    # Put all of the constraints together in one specification
    specification = on_any_stone.always(0,T) & \
                    reach_target.eventually(0,T)
    
    return specification, stones

def plot_stepping_stones_scenario(stones):
    ax = plt.gca()
    n_stones = len(stones)

    # Add rectangles for the stones
    for i in range(n_stones):
        if i < n_stones-1:  # ordinary stepping stones are orange
            ax.add_patch(make_rectangle_patch(*stones[i], color='orange', alpha=0.5, zorder=-1))
        else:  # the target is green
            ax.add_patch(make_rectangle_patch(*stones[i], color='g', alpha=0.5, zorder=-1))

    # set the field of view
    ax.set_xlim((0,10))
    ax.set_ylim((0,10))
    ax.set_aspect('equal')
