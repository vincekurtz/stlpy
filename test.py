#!/usr/bin/env python

##
# 
# Design and solve a simple specification, used for 
# debugging, etc. 
#
##

import numpy as np
import matplotlib.pyplot as plt
from scenarios.common import *
from scenarios.reach_avoid import *
from scenarios.either_or import *
from scenarios.random_multitarget import *
from solvers import SPPMICPSolver, MICPSolver, GradientSolver, PerspectiveMICPSolver

# Specification Parameters
goal_bounds = (7,8,8,9)     # (xmin, xmax, ymin, ymax)
obstacle_bounds = (3,5,4,6)
T = 10

# The "big-M" constant used for mixed-integer encoding
M = 1000

# Control bounds
u_min = -0.5
u_max = 0.5
control_bounded = inside_rectangle_formula((u_min,u_max,u_min,u_max), 4, 5, 6, name="control_bound")

# Goal Reaching
at_goal = inside_rectangle_formula(goal_bounds, 0, 1, 6, name="at_goal")

# Obstacle Avoidance
not_at_obstacle = outside_rectangle_formula(obstacle_bounds, 0, 1, 6, name="obstacle_avoidance")

# Velocity bounded
v_min = -1.0
v_max = 1.0
velocity_bounded = inside_rectangle_formula((v_min, v_max, v_min, v_max), 2, 3, 6, name="velocity_bound")

# Workspace boundaries
x_min = 0; x_max = 10;
y_min = 0; y_max = 10;
in_workspace = inside_rectangle_formula((x_min, x_max, y_min, y_max), 0, 1, 6, name="in_workspace")

# Put all of the constraints together in one specification
spec = control_bounded.always(0,T) & \
       velocity_bounded.always(0,T) & \
       in_workspace.always(0,T) & \
       not_at_obstacle.always(0,T) & \
       at_goal.eventually(0, T)
#       not_at_obstacle.until(at_goal, 0, T)

# DEBUG: more complicated specification
#spec, obstacles, targets = random_multitarget_specification(2, 2, 1, T, seed=0)
#plot_random_multitarget_scenario(obstacles,targets)
#plt.show(block=False)

plt.figure()

# System parameters
A = np.block([[1,0,1,0],
              [0,1,0,1],
              [0,0,1,0],
              [0,0,0,1]])
B = np.block([[0,0],
              [0,0],
              [1,0],
              [0,1]])

# Specify any additional running cost
Q = 1e-1*np.diag([0,0,1,1])   # just penalize high velocities
R = 1e-1*np.eye(2)

# Initial state
x0 = np.array([1.0,2.0,0,0])

# Solve for the system trajectory
solver = SPPMICPSolver(spec, A, B, Q, R, x0, T, relaxed=False)
#solver = PerspectiveMICPSolver(spec, A, B, Q, R, x0, T, relaxed=False)
#solver = MICPSolver(spec, A, B, Q, R, x0, T, M, relaxed=False)
#solver = GradientSolver(spec, A, B, Q, R, x0, T)
#solver.plot_partitions()

x, u = solver.Solve()

#solver.animate_partition_sequence()

if x is not None:
    # Set up a plot
    ax = plt.gca()

    # Make and add rectangular patches
    obstacle = make_rectangle_patch(*obstacle_bounds, color='red', alpha=0.5)
    goal = make_rectangle_patch(*goal_bounds, color='green', alpha=0.5)
    ax.add_patch(obstacle)
    ax.add_patch(goal)

    # set the field of view
    ax.set_xlim((0,10))
    ax.set_ylim((0,10))
    ax.set_aspect('equal')

    # Show the solution
    #solver.plot_relaxed_solution(show=False)
    plt.scatter(*x[:2,:])
    plt.show()
