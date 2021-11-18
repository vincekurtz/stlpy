#!/usr/bin/env python

##
# 
# Set up, solve, and plot the solution for a more complex
# reach-avoid problem involing multiple obstacles and multiple
# possible goals. 
#
##

import numpy as np
import matplotlib.pyplot as plt
from scenarios.random_multitarget import * 
from systems import LinearSystem
from solvers import *

# Specification Parameters
num_obstacles = 5
num_groups = 5
targets_per_group = 2
T = 20

# The "big-M" constant used for mixed-integer encoding
M = 1000

# Create the specification
spec, obstacles, targets = random_multitarget_specification(
        num_obstacles, num_groups, targets_per_group, T, seed=0)

# System dynamics
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
x0 = np.array([2.0,2.0,0,0])

# Solve for the system trajectory
solver = KnitroLCPSolver(spec, sys, x0, T)
x, u = solver.Solve()

if x is not None:
    # Plot the solution
    plot_random_multitarget_scenario(obstacles, targets)
    plt.scatter(*x[:2,:])
    plt.show()
