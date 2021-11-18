from solvers.drake.drake_base import DrakeSTLSolver
from STL import STLPredicate
import numpy as np
from pydrake.all import (MathematicalProgram, 
                         GurobiSolver, 
                         MosekSolver, 
                         eq, le, ge)

class DrakeMICPSolver(DrakeSTLSolver):
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
    
    .. note::

        This class implements the algorithm described in

        Raman V, et al. 
        *Model predictive control with signal temporal logic specifications*. 
        IEEE Conference on Decision and Control, 2014

    .. warning::

        Drake must be compiled from source to support Gurobi and Mosek MICP solvers.
        See `<https://drake.mit.edu/from_source.html>`_ for more details.

    :param spec:    An :class:`.STLFormula` describing the specification.
    :param sys:     A :class:`.LinearSystem` describing the system dynamics.
    :param x0:      A ``(n,1)`` numpy matrix describing the initial state.
    :param T:       A positive integer fixing the total number of timesteps :math:`T`.
    :param M:       A large positive scalar used to rewrite ``min`` and ``max`` as
                    mixed-integer constraints.
    :param relaxed: (optional) A boolean indicating whether to solve
                    a convex relaxation of the problem. Default is ``False``.
    """
    def __init__(self, spec, sys, x0, T, M, relaxed=False):
        assert M > 0, "M should be a (large) positive scalar"
        super().__init__(spec, sys, x0, T)
        self.M = M

        # Choose which solver to use
        #self.solver = GurobiSolver()
        self.solver = MosekSolver()

        # Create optimization variables
        self.y = self.mp.NewContinuousVariables(self.sys.p, self.T, 'y')
        self.x = self.mp.NewContinuousVariables(self.sys.n, self.T, 'x')
        self.u = self.mp.NewContinuousVariables(self.sys.m, self.T, 'u')
        self.rho = self.mp.NewContinuousVariables(1,'rho')[0]

        # Flag for whether to use a convex relaxation
        self.convex_relaxation = relaxed

        # Add cost and constraints to the optimization problem
        self.AddDynamicsConstraints()
        self.AddRobustnessCost()
        self.AddSTLConstraints()
        self.AddRobustnessConstraint()

    def AddControlBounds(self, u_min, u_max):
        for t in range(self.T):
            self.mp.AddConstraint(le(
                self.u[:,t], u_max
            ))
            self.mp.AddConstraint(ge(
                self.u[:,t], u_min
            ))
    
    def AddStateBounds(self, x_min, x_max):
        for t in range(self.T):
            self.mp.AddConstraint(le(
                self.x[:,t], x_max
            ))
            self.mp.AddConstraint(ge(
                self.x[:,t], x_min
            ))
    
    def AddQuadraticCost(self, Q, R):
        for t in range(self.T):
            self.mp.AddCost( self.x[:,t].T@Q@self.x[:,t] + self.u[:,t].T@R@self.u[:,t] )
    
    def AddRobustnessConstraint(self, rho_min=0.0):
        self.mp.AddConstraint( self.rho >= rho_min )

    def Solve(self, verbose=False, presolve=True):
        # Print out some solver data
        num_continuous_variables, num_binary_variables = self.GetVariableData()
        print("Solving MICP with")
        print("    %s binary variables" % num_binary_variables)
        print("    %s continuous variables" % num_continuous_variables)

        #if verbose:
        #    self.mp.SetSolverOption(solver.solver_id(), "OutputFlag", 1)
        #if not presolve:
        #    self.mp.SetSolverOption(solver.solver_id(), "Presolve", 0)

        res = self.solver.Solve(self.mp)

        solve_time = res.get_solver_details().optimizer_time
        print("Solve time: ", solve_time)

        if res.is_success():
            x = res.GetSolution(self.x)
            u = res.GetSolution(self.u)

            y = np.vstack([x,u])
            rho = self.spec.robustness(y,0)
            print("Optimal Cost: ", res.get_optimal_cost())
            print("Optimal robustness: ", rho[0])
        else:
            print("No solution found")
            x = None
            u = None

        return (x,u)

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

    def AddRobustnessCost(self):
        """
        Add the STL Robustness cost

            min -rho(y)

        to the optimization problem.
        """
        self.mp.AddCost(-self.rho)

    def AddSTLConstraints(self):
        """
        Add the STL constraints

            (x,u) |= specification

        to the optimization problem, via the recursive introduction
        of binary variables for all subformulas in the specification.
        """
        # Add a binary variable which takes a value of 1 only 
        # if the overall specification is satisfied.
        z_spec = self.NewBinaryVariables(1)
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
        if isinstance(formula, STLPredicate):
            # a.T*y - b + (1-z)*M >= rho
            y = self.y[:,t]
            self.mp.AddLinearConstraint(ge(
                formula.a.T@y - formula.b + (1-z)*self.M, self.rho
            ))
        
        # We haven't reached the bottom of the tree, so keep adding
        # boolean constraints recursively
        else:
            if formula.combination_type == "and":
                for i, subformula in enumerate(formula.subformula_list):
                    z_sub = self.NewBinaryVariables(1)
                    t_sub = formula.timesteps[i]   # the timestep at which this formula 
                                                   # should hold
                    self.AddSubformulaConstraints(subformula, z_sub, t+t_sub)
                    self.mp.AddConstraint( z[0] <= z_sub[0] )

            else:  # combination_type == "or":
                z_subs = []
                for i, subformula in enumerate(formula.subformula_list):
                    z_sub = self.NewBinaryVariables(1)
                    t_sub = formula.timesteps[i]
                    z_subs.append(z_sub)
                    self.AddSubformulaConstraints(subformula, z_sub, t+t_sub)

                # z <= sum(z_subs)
                A = np.hstack([1,-np.ones(len(z_subs))])[np.newaxis]
                lb = -np.array([np.inf])
                ub = np.array([0])
                vars = np.vstack([z,z_subs])
                self.mp.AddLinearConstraint(A=A, lb=lb, ub=ub, vars=vars)
