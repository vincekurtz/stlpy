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
from stlpy.benchmarks import NarrowPassage
from stlpy.solvers import *

# Specification Parameters
T = 25

# Create the specification and define the dynamics
scenario = NarrowPassage(T)
spec = scenario.GetSpecification()
spec.simplify()
sys = scenario.GetSystem()

# Specify any additional running cost (this helps the numerics in
# a gradient-based method)
Q = 1e-1*np.diag([0,0,1,1])   # just penalize high velocities
R = 1e-1*np.eye(2)

# Initial state
x0 = np.array([3.0,3.6,0,0])

# Specify a solution method
#solver = GurobiMICPSolver(spec, sys, x0, T, robustness_cost=True)
solver = DrakeMICPSolver(spec, sys, x0, T, robustness_cost=True, presolve=False)
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
