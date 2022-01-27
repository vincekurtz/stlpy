#!/usr/bin/env python

##
#
# Solving the Towers of Hanoi puzzle using
# STL.
#
##

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from STL import STLFormula, STLPredicate
from systems import LinearSystem
from solvers import DrakeMICPSolver, DrakeSos1Solver

######################################
# System definition
######################################
# We represent each ring with its horizontal and
# vertical position xr = [px,py]. The total state is the positions
# of all rings x = [x1,x2,x3,...]. 
#
# Control input u = [u1,u2,...] is similarly composed of 
# velocities ur = [vx,vy] for each ring. 

# Ring sizes
rh = 0.1   # height
rw = 0.5   # width

# The control input is change total position of each ring, and output
# is both position and velocity y = [x,u]
A = np.eye(2)
B = np.eye(2)
C = np.vstack([np.eye(2), np.zeros((2,2))])
D = np.vstack([np.zeros((2,2)), np.eye(2)])
sys = LinearSystem(A,B,C,D)

# Initial state
x0 = np.array([1.0,rh/2])

# Cost function penalizes large inputs
Q = np.zeros((2,2))
R = np.eye(2)

# Time horizon (max number of control actions)
T = 10

# Workspace boundaries
u_min = np.array([-1,-1])
u_max = np.array([1,1])
x_min = np.array([0,0])
x_max = np.array([2.5,2])

#######################################
# STL Specification
#######################################
# The STL specification constraints both the (approximate) dynamics
# of the system (e.g., only one ring can move at a time) as well as 
# the rules of the game (e.g., we can only stack smaller rings on larger ones)
eps = 1e-2   # small constant so that we can use strict > and <

# Define some basic predicates
no_x_movement = STLPredicate([0,0,1,0],[0]) & STLPredicate([0,0,-1,0],[0])  # vx <= 0 & vx >= 0
no_y_movement = STLPredicate([0,0,0,1],[0]) & STLPredicate([0,0,0,-1],[0])  # vy <= 0 & vy >= 0
no_movement = no_x_movement & no_y_movement

x_movement = STLPredicate([0,0,1,0],[eps]) | STLPredicate([0,0,-1,0],[eps]) # |vx| > 0
y_movement = STLPredicate([0,0,0,1],[eps]) | STLPredicate([0,0,0,-1],[eps]) # |vy| > 0
movement = x_movement | y_movement

on_ground = STLPredicate([0,-1,0,0],[-rh/2])  # py <= minimum height

on_first_pole = STLPredicate([1,0,0,0],[0]) & STLPredicate([-1,0,0,0],[0])   # px = 0
on_second_pole = STLPredicate([1,0,0,0],[1]) & STLPredicate([-1,0,0,0],[-1]) # px = 1
on_third_pole = STLPredicate([1,0,0,0],[2]) & STLPredicate([-1,0,0,0],[-2]) # px = 2
on_a_pole = on_first_pole | on_second_pole | on_third_pole

above_poles = STLPredicate([0,1,0,0],[1.2])  # py >= 1.2
below_poles = above_poles.negation()

# Eventually all rings must reach the third pole
reach_third_pole = (on_third_pole & no_movement).eventually(0,T)

# Rings cannot move in the x-direction unless they exceed a certain height
no_move_below_poles = no_x_movement | above_poles
never_move_below_poles = no_move_below_poles.always(0,T)

# Rings need to be on one of the poles if they're below a certain height
always_on_a_pole = (above_poles | on_a_pole).always(0,T)

# Can move horizontally or vertically, but not both
move_one_direction = (no_x_movement | no_y_movement).always(0,T)

# If a ring isn't moving, it must be on the ground
stop_on_ground = (movement | on_ground).always(0,T)

# Putting it all together
movement_rules = never_move_below_poles & \
                 move_one_direction & \
                 stop_on_ground & \
                 always_on_a_pole

game_rules = reach_third_pole

spec = movement_rules & game_rules

#######################################
# Solution visualization
#######################################
# Given a solution, we'd like to show a nice animation of how the blocks move

def plot_solution(x, save_fname=None):
    """
    Given a solution x, create a matplotlib animation of the solution.

    Args:
        x:          The (2*num_rings, T) numpy array representing 
                    the optimal positions of the rings.
        same_fname: (optional) filename for saving the animation. Doesn't
                    save if not provided.
    """
    # set up axes
    fig = plt.figure()
    ax = plt.gca()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_ylim((-0.1,2))
    ax.set_xlim((-0.5,2.5))

    # Poles
    plt.plot([0,0],[0,1],'k',linewidth=5)
    plt.plot([1,1],[0,1],'k',linewidth=5)
    plt.plot([2,2],[0,1],'k',linewidth=5)

    # Ground
    plt.fill_between([-100,100],[-10,0],color='k')

    # Rings
    r1 = plt.Rectangle([0,0],rw,rh, color='red')
    ax.add_patch(r1)

    def data_gen():
        # Generate data that gets sent to update the animation
        gen_list = (x[:,t] for t in range(x.shape[1]))
        return gen_list
    
    def update(data):
        # Update the animation based on data
        px, py = data
        r1.set_xy([px-rw/2,py-rh/2])


    ani = FuncAnimation(fig, update, data_gen)

    plt.show()

#######################################
# Solve the puzzle!
#######################################
solver = DrakeMICPSolver(spec, sys, x0, T, robustness_cost=True)
#solver = DrakeSos1Solver(spec, sys, x0, T, robustness_cost=False)
solver.AddQuadraticCost(Q,R)
solver.AddControlBounds(u_min, u_max)
solver.AddStateBounds(x_min, x_max)

x, u, _, _ = solver.Solve()

if x is not None:
    print(x)
    print(u)
    plot_solution(x)
