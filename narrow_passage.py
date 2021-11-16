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
from scenarios.narrow_passage import narrow_passage_specification, plot_narrow_passage_scenario
from solvers import *

# Specification Parameters
T = 20

# The "big-M" constant used for mixed-integer encoding
M = 1000

# Create the specification
spec = narrow_passage_specification(T)

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
x0 = np.array([2.0,2.0,0,0])

# Solve for the system trajectory
#solver = MICPSolver(spec, A, B, Q, R, x0, T, M)
#solver = GurobiMICPSolver(spec, A, B, x0, T, M)
solver = KnitroLCPSolver(spec, A, B, x0, T)
#solver = PerspectiveMICPSolver(spec, A, B, Q, R, x0, T)
#solver = GradientSolver(spec, A, B, Q, R, x0, T)
x, u = solver.Solve()

if x is not None:
    # Plot the solution
    plot_narrow_passage_scenario()
    plt.scatter(*x[:2,:])
    plt.show()
