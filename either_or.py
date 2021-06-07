#!/usr/bin/env python

##
# 
# Set up, solve, and plot the solution for the "reach-avoid"
# scenario, where the robot must reach one of two targets before
# reaching the goal. 
#
##

import numpy as np
import matplotlib.pyplot as plt
from scenarios.either_or import either_or_specification, plot_either_or_scenario
from solvers import MICPSolver, GradientSolver, PerspectiveMICPSolver, SPPMICPSolver

# Specification Parameters
goal = (7,8,8,9)     # (xmin, xmax, ymin, ymax)
target_one = (1,2,6,7)
target_two = (7,8,4.5,5.5)
obstacle = (3,5,4,6)
T = 22

# The "big-M" constant used for mixed-integer encoding
M = 1000

# Create the specification
spec = either_or_specification(goal, target_one, target_two, obstacle, T)

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
x0 = np.array([1.0,1.0,0,0])

# Solve for the system trajectory
solver = MICPSolver(spec, A, B, Q, R, x0, T, M, relaxed=False)
#solver = PerspectiveMICPSolver(spec, A, B, Q, R, x0, T, relaxed=True)
#solver = SPPMICPSolver(spec, A, B, Q, R, x0, T, relaxed=False)
#solver.plot_partitions()
#solver = GradientSolver(spec, A, B, Q, R, x0, T)
x, u = solver.Solve(verbose=True)

if x is not None:
    # Plot the solution
    plot_either_or_scenario(goal, target_one, target_two, obstacle)
    plt.scatter(*x[:2,:])
    plt.show()
