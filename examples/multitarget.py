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
from pySTL.benchmarks.random_multitarget import *
from pySTL.systems import DoubleIntegrator
from pySTL.solvers import *

# Specification Parameters
num_obstacles = 1
num_groups = 5
targets_per_group = 2
T = 15

# Create the specification
spec, obstacles, targets = random_multitarget_specification(
        num_obstacles, num_groups, targets_per_group, T, seed=0)

# System dynamics
sys = DoubleIntegrator(2)

# Specify any additional running cost (this helps the numerics in
# a gradient-based method)
Q = 1e-1*np.diag([0,0,1,1])   # just penalize high velocities
R = 1e-1*np.eye(2)

# Initial state
x0 = np.array([5.0,2.0,0,0])

# Define the solver
#solver = GurobiMICPSolver(spec, sys, x0, T, robustness_cost=True)
#solver = DrakeMICPSolver(spec, sys, x0, T, robustness_cost=True)
solver = DrakeSos1Solver(spec, sys, x0, T, robustness_cost=True)

# Set bounds on state and control variables
u_min = np.array([-0.5,-0.5])
u_max = np.array([0.5, 0.5])
x_min = np.array([0.0, 0.0, -1.0, -1.0])
x_max = np.array([10.0, 10.0, 1.0, 1.0])
solver.AddControlBounds(u_min, u_max)
solver.AddStateBounds(x_min, x_max)

# Add quadratic running cost (optional)
solver.AddQuadraticCost(0.01*Q,0.01*R)

# Solve the optimization problem
x, u, _, _ = solver.Solve()

if x is not None:
    # Plot the solution
    plot_random_multitarget_scenario(obstacles, targets)
    plt.scatter(*x[:2,:])
    plt.show()
