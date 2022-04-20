from .drake_base import DrakeSTLSolver
from ...STL import LinearPredicate, NonlinearPredicate
import numpy as np
import time
from pydrake.all import (GurobiSolver, MosekSolver, ClpSolver,
                         SolverOptions, CommonSolverOption,
                         eq, le, ge)
from pydrake.solvers.branch_and_bound import MixedIntegerBranchAndBound

class DrakeMICPSolver(DrakeSTLSolver):
    """
    Given an :class:`.STLFormula` :math:`\\varphi` and a :class:`.LinearSystem`,
    solve the optimization problem

    .. math::

        \min & -\\rho^{\\varphi}(y_0,y_1,\dots,y_T) + \sum_{t=0}^T x_t^TQx_t + u_t^TRu_t

        \\text{s.t. } & x_0 \\text{ fixed}

        & x_{t+1} = A x_t + B u_t

        & y_{t} = C x_t + D u_t

        & \\rho^{\\varphi}(y_0,y_1,\dots,y_T) \geq 0

    using mixed-integer convex programming. This gives a globally optimal
    solution, but may be computationally expensive for long and complex specifications.

    .. note::

        This class implements the algorithm described in

        Belta C, et al.
        *Formal methods for control synthesis: an optimization perspective*.
        Annual Review of Control, Robotics, and Autonomous Systems, 2019.
        https://dx.doi.org/10.1146/annurev-control-053018-023717.

    .. warning::

        Drake must be compiled from source to support the Gurobi MICP solver.
        See `<https://drake.mit.edu/from_source.html>`_ for more details. 

        Drake's naive branch-and-bound solver does not require Gurobi or Mosek, and
        can be used with the ``bnb`` solver option, but this tends to be very slow. 

    :param spec:            An :class:`.STLFormula` describing the specification.
    :param sys:             A :class:`.LinearSystem` describing the system dynamics.
    :param x0:              A ``(n,1)`` numpy matrix describing the initial state.
    :param T:               A positive integer fixing the total number of timesteps :math:`T`.
    :param M:               (optional) A large positive scalar used to rewrite ``min`` and ``max`` as
                            mixed-integer constraints. Default is ``1000``.
    :param robustness_cost: (optional) Boolean flag for adding a linear cost to maximize
                            the robustness measure. Default is ``True``.
    :param solver:          (optional) String describing the solver to use. Must be one
                            of 'gurobi', 'mosek', or 'bnb'.
    :param presolve:        (optional) A boolean indicating whether to use gurobi's
                            presolve routines. Only affects the gurobi solver. Default is ``True``.
    :param verbose:         (optional) A boolean indicating whether to print detailed
                            solver info. Default is ``True``.
    """
    def __init__(self, spec, sys, x0, T, M=1000, robustness_cost=True, 
            solver='gurobi', presolve=True, verbose=True):
        assert M > 0, "M should be a (large) positive scalar"
        super().__init__(spec, sys, x0, T, verbose)

        self.M = M
        self.presolve = presolve

        # Choose which solver to use
        if solver == 'gurobi':
            self.solver = GurobiSolver()
        elif solver == 'mosek':
            self.solver = MosekSolver()
        else:
            print("Using Naive Branch-and-Bound solver")
            self.solver = "bnb"

        # Add cost and constraints to the optimization problem
        self.AddDynamicsConstraints()
        self.AddSTLConstraints()
        self.AddRobustnessConstraint()
        if robustness_cost:
            self.AddRobustnessCost()

    def Solve(self):

        # Set solver options
        options = SolverOptions()
        if self.verbose:
            options.SetOption(CommonSolverOption.kPrintToConsole,1)
        if not self.presolve:
            options.SetOption(GurobiSolver.id(), "Presolve", 0)
        self.mp.SetSolverOptions(options)
        
        # Setup for naive branch and bound
        if self.solver == "bnb":
            bnb_solver = MixedIntegerBranchAndBound(self.mp, ClpSolver.id())
            st = time.time()
            status = bnb_solver.Solve()
            solve_time = time.time() - st
            success = True
            res = bnb_solver

        # Setup for Gurobi/Mosek
        else:
            res = self.solver.Solve(self.mp)
            success = res.is_success()
            solve_time = res.get_solver_details().optimizer_time

        if self.verbose:
            print("")
            print("Solve time: ", solve_time)

        if success:
            x = res.GetSolution(self.x)
            u = res.GetSolution(self.u)

            y = self.sys.C@x + self.sys.D@u
            rho = self.spec.robustness(y,0)[0]
            if self.verbose:
                print("Optimal robustness: ", rho)
        else:
            if self.verbose:
                print("No solution found")
            x = None
            u = None
            rho = -np.inf

        return (x,u, rho, solve_time)

    def AddDynamicsConstraints(self):
        """
        Add the constraints

            x_{t+1} = A@x_t + B@u_t
            x_0 = x0

        to the optimization problem.
        """
        # Initial condition
        self.mp.AddConstraint(eq( self.x[:,0], self.x0 ))

        # Dynamics
        for t in range(self.T-1):
            self.mp.AddConstraint(eq(
                self.x[:,t+1], self.sys.A@self.x[:,t] + self.sys.B@self.u[:,t]
            ))
            self.mp.AddConstraint(eq(
                self.y[:,t], self.sys.C@self.x[:,t] + self.sys.D@self.u[:,t]
            ))
        self.mp.AddConstraint(eq(
            self.y[:,self.T-1], self.sys.C@self.x[:,self.T-1] + self.sys.D@self.u[:,self.T-1]
        ))

    def AddSTLConstraints(self):
        """
        Add the STL constraints

            (x,u) |= specification

        to the optimization problem, via the recursive introduction
        of binary variables for all subformulas in the specification.
        """
        # Add a binary variable which takes a value of 1 only
        # if the overall specification is satisfied.
        z_spec = self.mp.NewContinuousVariables(1)
        self.mp.AddConstraint(eq( z_spec, 1 ))

        # Recursively traverse the tree defined by the specification
        # subformulas and add similar binary constraints.
        self.AddSubformulaConstraints(self.spec, z_spec, 0)

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
        if isinstance(formula, LinearPredicate):
            # a.T*y - b + (1-z)*M >= rho
            y = self.y[:,t]
            self.mp.AddLinearConstraint(ge(
                formula.a.T@y - formula.b + (1-z)*self.M, self.rho
            ))

            # By default, z is defined as a continuous variable. This
            # is a hack to make it binary. The redundant continuous
            # variable will be removed in presolve.
            tmp = self.mp.NewBinaryVariables(1)
            self.mp.AddConstraint(eq(tmp, z))

        elif isinstance(formula, NonlinearPredicate):
            raise TypeError("Mixed integer programming does not support nonlinear predicates")

        # We haven't reached the bottom of the tree, so keep adding
        # boolean constraints recursively
        else:
            z_subs = self.mp.NewContinuousVariables(len(formula.subformula_list),1)
            self.mp.AddConstraint(ge(z_subs, 0))

            if formula.combination_type == "and":
                self.mp.AddConstraint(le( z, z_subs ))
            else:  # combination_type == "or":
                self.mp.AddConstraint(le( z, sum(z_subs) ))

            for i, subformula in enumerate(formula.subformula_list):
                t_sub = formula.timesteps[i]
                self.AddSubformulaConstraints(subformula, z_subs[i], t+t_sub)
