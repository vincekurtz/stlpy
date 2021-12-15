from solvers.base import STLSolver
from STL import STLPredicate
import numpy as np

import cvxpy as cp
import ncvx as nc

import time

class AdmmSolver(STLSolver):
    """
    Given an :class:`.STLFormula` :math:`\\varphi` and a :class:`.LinearSystem`, 
    solve the optimization problem

    .. math:: 

        \max & \\rho^{\\varphi}(y_0,y_1,\dots,y_T)

        \\text{s.t. } & x_0 \\text{ fixed}

        & x_{t+1} = f(x_t, u_t) 

        & y_{t} = g(x_t, u_t)

        & \\rho^{\\varphi}(y_0,y_1,\dots,y_T) \geq 0

    Using (nonconvex) ADMM as a solver. 

    .. note::

        This class uses ADMM to solve a nonconvex program as described in

        Diamond S, et al.
        *A General System for Heuristic Solution of Convex Problems over Nonconvex Sets*
        Optimization Methods and Software, 2018.


    :param spec:            An :class:`.STLFormula` describing the specification.
    :param sys:             A :class:`.LinearSystem` describing the system dynamics.
    :param x0:              A ``(n,1)`` numpy matrix describing the initial state.
    :param T:               A positive integer fixing the total number of timesteps :math:`T`.
    :param robustness_cost: (optional) Boolean flag for adding a linear cost to maximize
                            the robustness measure. Default is ``True``.
    """
    def __init__(self, spec, sys, x0, T, robustness_cost=True):
        super().__init__(spec, sys, x0, T)
