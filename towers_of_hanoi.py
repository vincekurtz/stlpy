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
# of all rings x = [x1,x2,x3...]

# Ring sizes
rh = 0.1   # height
rw = 0.5   # width

# The control input is change total position of each ring
A = np.eye(2)
B = np.eye(2)
C = np.eye(2)
D = np.zeros((2,2))
sys = LinearSystem(A,B,C,D)

# Initial state
x0 = np.array([0,rh/2])

# Cost function penalizes large inputs
Q = np.zeros((2,2))
R = np.eye(2)

# Time horizon (max number of control actions)
T = 10

#######################################
# STL Specification
#######################################
# The STL specification constraints both the (approximate) dynamics
# of the system (e.g., only one ring can move at a time) as well as 
# the rules of the game (e.g., we can only stack smaller rings on larger ones)

# Eventually all rings must reach the third pole
ring_on_third_pole = STLPredicate([[1],[0]],[2])   # px >= 2
solve_puzzle = ring_on_third_pole.eventually(0,T)

# Putting it all together
spec = solve_puzzle

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
solver = DrakeMICPSolver(spec, sys, x0, T, robustness_cost=False)
solver.AddQuadraticCost(Q,R)

x, u, _, _ = solver.Solve()

if x is not None:
    print(x)
    plot_solution(x)
