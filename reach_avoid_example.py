#!/usr/bin/env python

##
# 
# Set up, solve, and plot the solution for a simple
# reach-avoid problem using gradient-based synthesis. 
#
##

import numpy as np
from scipy.optimize import minimize
from reach_avoid_scenario import reach_avoid_specification, plot_reach_avoid_scenario
from gradient_based_optimization import GradientSolver

# Specification Parameters
goal_bounds = (7,8,8,9)     # (xmin, xmax, ymin, ymax)
obstacle_bounds = (3,5,4,6)
T = 10

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

# Initial state
x0 = np.array([1.0,1.0,0,0])

# Solve for the system trajectory
solver = GradientSolver(spec, A, B, T, x0)
x, u = solver.Solve()

# Plot the solution
