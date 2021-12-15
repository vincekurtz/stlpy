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
    :param M:               (optional) A large positive scalar used to rewrite ``min`` and ``max`` as
                            mixed-integer constraints. Default is ``1000``.
    :param robustness_cost: (optional) Boolean flag for adding a linear cost to maximize
                            the robustness measure. Default is ``True``.
    """
    def __init__(self, spec, sys, x0, T, M = 1000, robustness_cost=True):
        assert M > 0, "M should be a (large) positive scalar"
        super().__init__(spec, sys, x0, T)
        self.M = float(M)

        # Create primary decision variables
        self.x = cp.Variable((self.sys.n,self.T))
        self.u = cp.Variable((self.sys.m,self.T))
        self.y = cp.Variable((self.sys.p,self.T))
        self.rho = cp.Variable(1)

        # Create list to hold constraints
        self.constraints = []

        # Set placeholder objective function
        self.objective = cp.Minimize(0)

        # Set up the optimization problem
        self.AddDynamicsConstraints()
        self.AddSTLConstraints()
        self.AddRobustnessConstraint()
        if robustness_cost:
            self.AddRobustnessCost()

    def AddControlBounds(self, u_min, u_max):
        for t in range(self.T):
            self.constraints.append( u_min <= self.u[:,t] )
            self.constraints.append( self.u[:,t] <= u_max )

    def AddStateBounds(self, x_min, x_max):
        for t in range(self.T):
            self.constraints.append( x_min <= self.x[:,t] )
            self.constraints.append( self.x[:,t] <= x_max )

    def AddDynamicsConstraints(self):
        # Initial condition
        self.constraints.append( self.x[:,0] == self.x0 )

        # Dynamics
        for t in range(self.T-1):
            self.constraints.append(
                    self.x[:,t+1] == self.sys.A@self.x[:,t] + self.sys.B@self.u[:,t] )
        for t in range(self.T):
            self.constraints.append(
                    self.y[:,t] == self.sys.C@self.x[:,t] + self.sys.D@self.u[:,t] )

    def AddQuadraticCost(self, Q, R):
        pass

    def AddRobustnessConstraint(self, rho_min=0.0):
        self.constraints.append( self.rho >= rho_min )

    def AddRobustnessCost(self):
        self.objective = cp.Maximize(self.rho)

    def AddSTLConstraints(self):
        """
        Add the STL constraints

            (x,u) |= specification

        to the optimization problem, via the recursive introduction
        of binary variables for all subformulas in the specification.
        """
        # Recursively traverse the tree defined by the specification
        # to add binary variables and constraints that ensure that
        # rho is the robustness value
        z_spec = cp.Variable(1,boolean=True)  
        #z_spec = nc.Boolean(1)
        self.AddSubformulaConstraints(self.spec, z_spec, 0)
        self.constraints.append( z_spec == 1 )

    def AddSubformulaConstraints(self, formula, z, t):
        """
        Given an STLFormula (formula) and a binary variable (z),
        add constraints to the optimization problem such that z
        takes value 1 only if the formula is satisfied (at time t). 

        If the formula is a predicate, this constraint uses the "big-M" 
        formulation

            A[x(t);u(t)] - b + (1-z)M >= 0,

        which enforces A[x;u] - b >= 0 if z=1, where (A,b) are the 
        linear constraints associated with this predicate. 

        If the formula is not a predicate, we recursively traverse the
        subformulas associated with this formula, adding new binary 
        variables z_i for each subformula and constraining

            z <= z_i  for all i

        if the subformulas are combined with conjunction (i.e. all 
        subformulas must hold), or otherwise constraining

            z <= sum(z_i)

        if the subformulas are combined with disjuction (at least one
        subformula must hold). 
        """
        # We're at the bottom of the tree, so add the big-M constraints
        if isinstance(formula, STLPredicate):
            # a.T*y - b + (1-z)*M >= rho
            self.constraints.append( 
                    formula.a.T@self.y[:,t] - formula.b + (1-z)*self.M  >= self.rho )
        
        # We haven't reached the bottom of the tree, so keep adding
        # boolean constraints recursively
        else:
            if formula.combination_type == "and":
                for i, subformula in enumerate(formula.subformula_list):
                    z_sub = cp.Variable(1,boolean=True)  
                    t_sub = formula.timesteps[i]   # the timestep at which this formula 
                                                   # should hold
                    self.AddSubformulaConstraints(subformula, z_sub, t+t_sub)
                    self.constraints.append( z <= z_sub )

            else:  # combination_type == "or":
                z_subs = []
                for i, subformula in enumerate(formula.subformula_list):
                    z_sub = cp.Variable(1,boolean=True)  
                    t_sub = formula.timesteps[i]
                    z_subs.append(z_sub)
                    self.AddSubformulaConstraints(subformula, z_sub, t+t_sub)

                self.constraints.append( z <= sum(z_subs) )

    def Solve(self):
        prob = cp.Problem(self.objective, self.constraints)
        st = time.time()
        #result = prob.solve(method="NC-ADMM", verbose=True)
        result = prob.solve(solver='GUROBI', verbose=True)
        solve_time = time.time() - st

        x = self.x.value
        u = self.u.value
        rho = self.rho.value

        print(result)
        print(prob.status)
        print(prob.solver_stats)
        print(solve_time)
        print(rho)

        return (x, u, rho, solve_time)

