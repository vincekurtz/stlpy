from scenarios.common import *
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

##
#
# Tools for setting up a simple specification where a robot must reach
# one of several potential goal regions at a given time. 
#
##

def fixed_endtime_specification(T):
    """
    Return an STLFormula that describes this scenario
    
    We'll assume that the robot has double integrator dynamics, i.e.,
    
        x = [px,py,pdx,pdy], u = [pddx, pddy]
    
    and that the output signal is given by y = [x;u].
    """
    t1 = (1,2,1,2)
    t2 = (7,8,7,8)

    # Reaching each individual target
    at_t1 = inside_rectangle_formula(t1, 0, 1, 6)
    at_t2 = inside_rectangle_formula(t2, 0, 1, 6)

    # Reaching any of the targets
    at_any_target = at_t1 | at_t2

    # Overall specification
    specification = at_any_target.eventually(T,T)

    return specification

def plot_fixed_endtime_scenario():
    ax = plt.gca()
    
    t1 = (1,2,1,2)
    t2 = (7,8,7,8)

    targets = [t1,t2]

    # Make and add rectangular patches
    for target in targets:
        patch = make_rectangle_patch(*target, color='blue', alpha=0.5)
        ax.add_patch(patch)

    # set the field of view
    ax.set_xlim((0,10))
    ax.set_ylim((0,10))
    ax.set_aspect('equal')
