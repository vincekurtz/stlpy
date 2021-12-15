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
from systems import LinearSystem
from solvers import *

# Specification Parameters
goal = (7,8,8,9)     # (xmin, xmax, ymin, ymax)
target_one = (1,2,6,7)
target_two = (7,8,4.5,5.5)
obstacle = (3,5,4,6)
T = 15

# Create the specification
spec = either_or_specification(goal, target_one, target_two, obstacle, T)

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

# Specify any additional running cost
Q = 1e-1*np.diag([0,0,1,1])   # just penalize high velocities
R = 1e-1*np.eye(2)

# Initial state
x0 = np.array([1.0,1.0,0,0])

# Solve for the system trajectory
#solver = GurobiMICPSolver(spec, sys, x0, T, robustness_cost=True)
solver = DrakeMICPSolver(spec, sys, x0, T, robustness_cost=True)
#solver = KnitroLCPSolver(spec, sys, x0, T, robustness_cost=False)
#solver = DrakeLCPSolver(spec, sys, x0, T, robustness_cost=False)
#solver = DrakeSmoothSolver(spec, sys, x0, T)
#solver.AddQuadraticCost(Q,R)
x, u, _, _ = solver.Solve()

if x is not None:
    # Plot the solution
    plot_either_or_scenario(goal, target_one, target_two, obstacle)
    plt.scatter(*x[:2,:])
    plt.show()
