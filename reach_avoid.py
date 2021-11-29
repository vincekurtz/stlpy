#!/usr/bin/env python

##
# 
# Set up, solve, and plot the solution for a simple
# reach-avoid problem, where the robot must avoid
# a rectangular obstacle before reaching a rectangular
# goal.
#
##

import numpy as np
import matplotlib.pyplot as plt
from scenarios.reach_avoid import reach_avoid_specification, plot_reach_avoid_scenario
from systems import LinearSystem
from solvers import *

# Specification Parameters
goal_bounds = (7,8,8,9)     # (xmin, xmax, ymin, ymax)
obstacle_bounds = (3,5,4,6)
T = 10

# The "big-M" constant used for mixed-integer encoding
M = 1000

# Create the specification
spec = reach_avoid_specification(goal_bounds, obstacle_bounds, T)

# Define the system
A = np.block([[1,0,1,0],
              [0,1,0,1],
              [0,0,1,0],
              [0,0,0,1]])
B = np.block([[0,0],
              [0,0],
              [1,0],
              [0,1]])
C = np.block([[np.eye(4)],
              [np.zeros((2,4))]])
D = np.block([[np.zeros((4,2))],
              [np.eye(2)]])

sys = LinearSystem(A,B,C,D)

# Specify any additional running cost (this helps the numerics in 
# a gradient-based method)
Q = 1e-1*np.diag([0,0,1,1])   # just penalize high velocities
R = 1e-1*np.eye(2)

# Initial state
x0 = np.array([1.0,2.0,0,0])

# Solve for the system trajectory
#solver = ScipyGradientSolver(spec, sys, Q, R, x0, T, method="powell")
#solver = GurobiMICPSolver(spec, sys, x0, T, M, robustness_cost=False)
#solver = GurobiLCPSolver(spec, sys, x0, T, robustness_cost=False)
solver = KnitroLCPSolver(spec, sys, x0, T, robustness_cost=False)
#solver = DrakeMICPSolver(spec, sys, x0, T, M, robustness_cost=False)
#solver = DrakeLCPSolver(spec, sys, x0, T, robustness_cost=True)
#solver = DrakeSmoothSolver(spec, sys, x0, T)

#solver.AddQuadraticCost(Q,R)

x, u, _, _ = solver.Solve()

if x is not None:
    # Plot the solution
    plot_reach_avoid_scenario(goal_bounds, obstacle_bounds)
    plt.scatter(*x[:2,:])
    plt.show()
