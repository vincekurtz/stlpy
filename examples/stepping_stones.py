#!/usr/bin/env python

##
#
# Set up, solve, and plot the solution for a reachability
# problem where the robot must navigate over a stepping
# stones in order to reach a goal.
#
##

import numpy as np
import matplotlib.pyplot as plt
from pySTL.benchmarks import SteppingStones
from pySTL.solvers import *

# Specification Parameters
num_stones = 15
T = 15

# Define the specification and system dynamics
scenario = SteppingStones(num_stones, T, seed=1)
spec = scenario.GetSpecification()
spec.simplify()
sys = scenario.GetSystem()

# Specify any additional running cost (this helps the numerics in
# a gradient-based method)
Q = 1e-1*np.diag([0,0,1,1])   # just penalize high velocities
R = 1e-1*np.eye(2)

# Initial state
x0 = np.array([2.0,1.3,0,0])

# Specify a solution strategy
#solver = GurobiMICPSolver(spec, sys, x0, T, robustness_cost=True)
solver = DrakeMICPSolver(spec, sys, x0, T, robustness_cost=True)
#solver = DrakeSos1Solver(spec, sys, x0, T, robustness_cost=True)

# Set bounds on state and control variables
u_min = np.array([-0.5,-0.5])
u_max = np.array([0.5, 0.5])
x_min = np.array([0.0, 0.0, -1.0, -1.0])
x_max = np.array([10.0, 10.0, 1.0, 1.0])
solver.AddControlBounds(u_min, u_max)
solver.AddStateBounds(x_min, x_max)

# Add quadratic running cost (optional)
solver.AddQuadraticCost(Q,R)

# Solve the optimization problem
x, u, _, _ = solver.Solve()

if x is not None:
    # Plot the solution
    ax = plt.gca()
    scenario.add_to_plot(ax)
    plt.scatter(*x[:2,:])
    plt.show()
