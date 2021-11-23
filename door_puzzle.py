#!/usr/bin/env python

##
# 
# Set up, solve, and plot the solution for the door-puzzle
# scenario, where a robot needs to pick up several keys 
# before reaching a goal region.
#
##

import numpy as np
import matplotlib.pyplot as plt
from scenarios.door_puzzle import door_puzzle_specification, plot_door_puzzle_scenario
from systems import LinearSystem
from solvers import *

# Specification Parameters
T = 30

# The "big-M" constant used for mixed-integer encoding
M = 1000

# Create the specification
spec = door_puzzle_specification(T)

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
x0 = np.array([3.0,3.0,0,0])

# Solve for the system trajectory
#solver = ScipyGradientSolver(spec, sys, Q, R, x0, T)
#solver = GurobiMICPSolver(spec, sys, x0, T, M, robustness_cost=False)
#solver = GurobiLCPSolver(spec, sys, x0, T, robustness_cost=False)
solver = KnitroLCPSolver(spec, sys, x0, T, robustness_cost=False)
#solver = DrakeMICPSolver(spec, sys, x0, T, M, robustness_cost=False)
#solver = DrakeLCPSolver(spec, sys, x0, T, robustness_cost=False)
#solver = DrakeSmoothSolver(spec, sys, x0, T)

#x_min = np.array([0,0,-1,-1])
#x_max = np.array([15,10,1,1])
#u_max = np.array([0.5,0.5])
#y_min = np.hstack([x_min,-u_max])
#y_max = np.hstack([x_max,u_max])
#solver.AddStateBounds(x_min, x_max)
#solver.AddControlBounds(-u_max, u_max)
#solver.AddOutputBounds(y_min, y_max)

x, u, _, _ = solver.Solve()

if x is not None:
    # Plot the solution
    plot_door_puzzle_scenario()
    plt.scatter(*x[:2,:])
    plt.show()
