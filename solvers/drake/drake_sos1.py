from solvers.drake.drake_micp import DrakeMICPSolver
from STL import STLPredicate
import numpy as np
from pydrake.all import (MathematicalProgram, 
                         GurobiSolver, MosekSolver, 
                         SolverOptions, CommonSolverOption,
                         AddLogarithmicSos1Constraint,
                         eq, le, ge)

class DrakeSos1Solver(DrakeMICPSolver):
    """
    Given an :class:`.STLFormula` :math:`\\varphi` and a :class:`.LinearSystem`, 
    solve the optimization problem

    .. math:: 

        \max ~& \\rho^{\\varphi}(y_0,y_1,\dots,y_T)

        \\text{s.t. } & x_0 \\text{ fixed}

        & x_{t+1} = f(x_t, u_t) 

        & y_{t} = g(x_t, u_t)

        & y_0,y_1,\dots,y_T \\vDash \\varphi

    using mixed-integer convex programming. This gives a globally optimal
    solution, but may be computationally expensive for long and complex specifications.

    This class uses the encoding described in

        Vielma J, et al. 
        *Modeling disjunctive constraints with a logarithmic number of binary variables and
        constraints*. Mathematical Programming, 2011.

    to use fewer binary variables, improving scalability to long specifications. 
    
    .. warning::

        Drake must be compiled from source to support Gurobi and Mosek MICP solvers.
        See `<https://drake.mit.edu/from_source.html>`_ for more details.

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
    """
    def __init__(self, spec, sys, x0, T, M=1000, robustness_cost=True, solver='gurobi', presolve=True):
        super().__init__(spec, sys, x0, T, M, robustness_cost=robustness_cost, solver=solver, presolve=presolve)

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
            y = self.y[:,t]
            self.mp.AddLinearConstraint(ge(
                formula.a.T@y - formula.b + (1-z)*self.M, self.rho
            ))

        # We haven't reached the bottom of the tree, so keep adding
        # boolean constraints recursively
        else:
            nz = len(formula.subformula_list)

            if formula.combination_type == "and":
                z_subs = self.mp.NewContinuousVariables(nz,1)
                self.mp.AddConstraint(le( z, z_subs ))

            else:  # combination_type == "or":
                lambda_, _ = AddLogarithmicSos1Constraint(self.mp, nz + 1)
                z_subs = lambda_[1:][np.newaxis].T
                self.mp.AddConstraint(eq( 1-z, lambda_[0] ))

            for i, subformula in enumerate(formula.subformula_list):
                t_sub = formula.timesteps[i]
                self.AddSubformulaConstraints(subformula, z_subs[i], t+t_sub)

