#!/usr/bin/env python

##
# 
# Set up, solve, and plot the solution for a simple
# reach-avoid problem using simple gradient-based synthesis. 
#
##

import numpy as np
import matplotlib.pyplot as plt
from reach_avoid_scenario import reach_avoid_specification, plot_reach_avoid_scenario
from gradient_based_optimization import GradientSolver

# Specification Parameters
goal_bounds = (7,8,8,9)     # (xmin, xmax, ymin, ymax)
obstacle_bounds = (3,5,4,6)
T = 20

# Create the specification
spec = reach_avoid_specification(goal_bounds, obstacle_bounds, T)

# System parameters
A = np.block([[1,0,1,0],
              [0,1,0,1],
              [0,0,1,0],
              [0,0,0,1]])
B = np.block([[0,0],
              [0,0],
              [1,0],
              [0,1]])

# Specify any additional running cost (this helps the numerics in 
# a gradient-based method)
Q = 1e-1*np.diag([0,0,1,1])   # just penalize high velocities
R = 1e-1*np.eye(2)

# Initial state
x0 = np.array([1.0,3.0,0,0])

# Solve for the system trajectory
solver = GradientSolver(spec, A, B, Q, R, x0, T)
x, u = solver.Solve()

if x is not None:
    # Plot the solution
    plot_reach_avoid_scenario(goal_bounds, obstacle_bounds)
    plt.scatter(*x[:2,:])
    plt.show()
