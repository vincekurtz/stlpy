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
import csv

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
    if solver.__name__=="DrakeSmoothSolver":
        s = solver(spec, sys, x0, T)
    else:
        s = solver(spec, sys, x0, T, robustness_cost=linear_cost)
   
    if quadratic_cost:
        # Specify the (optional) quadratic running cost ( x'Qx + u'Ru )
        Q = 1e-1*np.diag([0,0,1,1])   # just penalize high velocities
        R = 1e-1*np.eye(2)
        s.AddQuadraticCost(Q,R)

    # Get a solution
    x, u, rho, solve_time = s.Solve()
    success = x is not None

    return (success, rho, solve_time)

def run_timestep_tests(fname):
    """
    Run a sequence of tests for different solvers with different
    numbers of timesteps (on the same scenario) and save the results
    to a CSV file. 

    :param fname:   File name to save the results to.
    """
    f = open(fname,'w')
    writer = csv.writer(f,delimiter=',')

    data = ["Solver", "T", "n_obs", "n_group", "tpg", 
            "linear_cost", "quadratic_cost", "rho", "solve_time"]
    writer.writerow(data)

    solvers = [GurobiMICPSolver, KnitroLCPSolver, DrakeLCPSolver, DrakeSmoothSolver]
    Ts = [10,15,20,25,30,40,50,70,100]
    n_obs = 2
    n_group = 2
    tpg = 2
    linear_costs = [False, True]
    quadratic_costs = [False, True]
    for solver in solvers:
        for T in Ts:
            for linear_cost in linear_costs:
                for quadratic_cost in quadratic_costs:

                    if solver.__name__ == "DrakeSmoothSolver" and linear_cost == False:
                        # Smooth solver requires linear cost
                        pass

                    else:
                        # Run the experiment and write to the CSV file
                        success, rho, solve_time = get_solution(solver, n_obs, n_group, tpg, T, 
                                linear_cost=linear_cost, quadratic_cost=quadratic_cost)

                        data = [solver.__name__, T, n_obs, n_group, tpg, 
                                linear_cost, quadratic_cost, rho, solve_time]

                        writer.writerow(data)

    f.close()


if __name__=="__main__":
    run_timestep_tests("timestep_tests.csv")
