from stl_formula import STLFormula, STLPredicate
import matplotlib.pyplot as plt

##
#
# Set up a simple specification, which is to reach a goal region and avoid an obstacle. 
#
##

# Define the goal and obstacle regions as rectangles (could be polytopes in general,
# but this makes it simpler).
goal_bounds = (7,8,8,9)     # (xmin, xmax, ymin, ymax)
obstacle_bounds = (3,5,4,6)

# We'll assume that the robot has double integrator dynamics, i.e.,
#
#   x = [px,py,pdx,pdy], u = [pddx, pddy]
#
# and that the output signal is given by y = [x;u].
d = 6   # size of y = [x;u]

# Define control bound constraints
u_min = -0.2
u_max = 0.2

u1_above_min = STLPredicate([[0,0,0,0,1,0]],[u_min])
u2_above_min = STLPredicate([[0,0,0,0,0,1]],[u_min])
u1_below_max = STLPredicate([[0,0,0,0,-1,0]],[-u_max])
u2_below_max = STLPredicate([[0,0,0,0,-1]],[-u_max])

control_bounded = u1_above_min.conjunction(u2_above_min).conjunction(u1_below_max).conjunction(u2_below_max)

# Define the goal reaching specification

def plot_reach_avoid_scenario(goal_bounds, obstacle_bounds):
    pass





