#!/usr/bin/env python

##
#
# Tools for running automated scalability comparisons 
# for different STL synthesis appraoches.
#
##

import numpy as np
from scenarios.random_multitarget import *
from systems import LinearSystem
from solvers import *

def get_solution(solver, n_obs, n_group, tpg, T, linear_cost=True, quadratic_cost=False, seed=0):
    """
    For a given solver, return solution data for a random multitarget
    scenario with the given parameters.

    :param solver:          An uninstantiated :class:`.STLSolver`. Used to find a 
                            satisfying trajectory.
    :param n_obs:           The number of obstacles to add to the scenario.
    :param n_group:         The number of different target groups to add to the scenario.
    :param tpg:             The number of target regions per target group.
    :param T:               The total number of timesteps in the trajectory
    :param linear_cost:     (optional) Boolean flag for inlcuding a linear cost
                            to maximize the STL robustness score. Default is ``True``.
    :param quadratic_cost:  (optional) Boolean flag for inlcuding a quadratic running
                            cost. Default is ``True``.
    :param seed:            (optional) The seed for the psuedorandom number generator.

    :return success:    Boolean flag for whether the solver found a solution.
    :return rho:        Optimal robustness value.
    :return solve_time: Solver time in seconds.
    """
    # set up the system dynamics
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

    # Create the specification
    spec, _, _ = random_multitarget_specification(
            n_obs, n_group, tpg, T, seed=seed)


    # Initial state
    x0 = np.array([2.0,2.0,0,0])

    # Solver setup
    s = solver(spec, sys, x0, T, robustness_cost=linear_cost)
   
    if quadratic_cost:
        # Specify the (optional) quadratic running cost ( x'Qx + u'Ru )
        Q = 1e-1*np.diag([0,0,1,1])   # just penalize high velocities
        R = 1e-1*np.eye(2)
        s.AddQuadraticRunningCost(Q,R)

    # Get a solution
    x, u, rho, solve_time = s.Solve()
    success = x is not None

    return (success, rho, solve_time)

if __name__=="__main__":
    solver = GurobiMICPSolver

    n_obs = 2
    n_group = 2
    tpg = 2
    T = 20

    success, rho, solve_time = get_solution(solver, n_obs, n_group, tpg, T)

    print(success, rho, solve_time)
