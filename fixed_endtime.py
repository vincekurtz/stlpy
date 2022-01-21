#!/usr/bin/env python

##
# 
# Set up, solve, and plot the solution for a simple
# reachability problem, where the robot must be at 
# one of several target locations at the final timestep.
#
##

import numpy as np
import matplotlib.pyplot as plt
from scenarios.fixed_endtime import fixed_endtime_specification, plot_fixed_endtime_scenario
from systems import LinearSystem
from solvers import *

# Specification Parameters
T = 20

# Create the specification
spec = fixed_endtime_specification(T)
spec.simplify()

# Define the system
# TODO: make double integrator abstraction in scenarios.common or systems
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
x0 = np.array([5.0,5.0,0,0])

# Define the solver
#solver = ScipyGradientSolver(spec, sys, Q, R, x0, T, method="powell")
#solver = GurobiMICPSolver(spec, sys, x0, T, robustness_cost=True)
#solver = GurobiLCPSolver(spec, sys, x0, T, robustness_cost=True)
#solver = KnitroLCPSolver(spec, sys, x0, T, robustness_cost=False)
solver = DrakeMICPSolver(spec, sys, x0, T, robustness_cost=True)
#solver = DrakeTestSolver(spec, sys, x0, T, robustness_cost=True)
#solver = DrakeSos1Solver(spec, sys, x0, T, robustness_cost=True)
#solver = AdmmSolver(spec, sys, x0, T, robustness_cost=True)
#solver = DrakeLCPSolver(spec, sys, x0, T, robustness_cost=False)
#solver = DrakeSmoothSolver(spec, sys, x0, T)

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
    plot_fixed_endtime_scenario()
    plt.scatter(*x[:2,:])
    plt.show()
