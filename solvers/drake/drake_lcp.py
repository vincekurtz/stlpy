from solvers.drake.drake_base import DrakeSTLSolver
from STL import STLPredicate
import numpy as np

from pydrake.all import MathematicalProgram, eq, le, ge
from pydrake.solvers.all import IpoptSolver, SnoptSolver
from pydrake.solvers.nlopt import NloptSolver

from pydrake.all import SolverOptions, CommonSolverOption, SolverType

import time

class DrakeLCPSolver(DrakeSTLSolver):
    """
    Given an :class:`.STLFormula` :math:`\\varphi` and a :class:`.LinearSystem`, 
    solve the optimization problem

    .. math:: 

        \max ~& \\rho^{\\varphi}(y_0,y_1,\dots,y_T)

        \\text{s.t. } & x_0 \\text{ fixed}

        & x_{t+1} = f(x_t, u_t) 

        & y_{t} = g(x_t, u_t)

        & \\rho^{\\varphi}(y_0,y_1,\dots,y_T) \geq 0

    where :math:`\\rho^{\\varphi}` is defined using linear complementarity constraints
    rather than mixed-integer constraints. We then use one of the general
    nonlinear solvers availible in Drake to find a locally optimal solution. 

    :param spec:            An :class:`.STLFormula` describing the specification.
    :param sys:             A :class:`.LinearSystem` describing the system dynamics.
    :param x0:              A ``(n,1)`` numpy matrix describing the initial state.
    :param T:               A positive integer fixing the total number of timesteps :math:`T`.
    :param robustness_cost: (optional) Boolean flag for adding a linear cost to maximize
                            the robustness measure. Default is ``True``.
    """
    def __init__(self, spec, sys, x0, T, robustness_cost=True):
        print("Setting up optimization problem...")
        st = time.time()  # for computing setup time
        
        super().__init__(spec, sys, x0, T)

        # Choose a solver
        #self.solver = IpoptSolver()
        self.solver = SnoptSolver()

        # Set some solver options
        options = SolverOptions()
        options.SetOption(CommonSolverOption.kPrintToConsole, 1)  # SNOPT doesn't support this
        #options.SetOption(CommonSolverOption.kPrintFileName, "snopt_out.txt")
        options.SetOption(SnoptSolver().solver_id(), "Elastic weight", 1e4)  # activates elastic
                                                                             # mode, which is
                                                                             # apparently critical
        self.mp.SetSolverOptions(options)

        # Add cost and constraints to the optimization problem
        self.AddDynamicsConstraints()
        self.AddSTLConstraints()
        self.AddRobustnessConstraint()
        if robustness_cost:
            self.AddRobustnessCost()
        
        print(f"Setup complete in {time.time()-st} seconds.")
    
    def AddDynamicsConstraints(self):
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


    def Solve(self):
        # Local solvers tend to be sensitive to the initial guess
        np.random.seed(0)
        initial_guess = np.random.normal(size=self.mp.initial_guess().shape)

        st = time.time()
        res = self.solver.Solve(self.mp, initial_guess=initial_guess)
        solve_time = time.time() - st

        if res.is_success():
            print("\nOptimal Solution Found!\n")
            x = res.GetSolution(self.x)
            u = res.GetSolution(self.u)

            # Report solve time and robustness
            y = np.vstack([x,u])
            rho = self.spec.robustness(y,0)
            print("Solve time: ", solve_time)
            print("Optimal robustness: ", rho[0])

        else:
            print("\nNo solution found.\n")
            x = None
            u = None

        return (x,u)

    def AddSTLConstraints(self):
        """
        Add the STL constraints

            (x,u) |= specification

        to the optimization problem, via the recursive introduction
        of binary variables for all subformulas in the specification.
        """
        # Recursively traverse the tree defined by the specification
        # to add constraints that define the STL robustness score
        self.AddSubformulaConstraints(self.spec, self.rho, 0)

    def AddSubformulaConstraints(self, formula, rho, t):
        """
        Given an STLFormula (formula) and a continuous variable (rho),
        add constraints to the optimization problem such that rho is
        the robustness score for that formula.

        If the formula is a predicate (Ay-b>=0), this means that

            rho = A[x(t);u(t)] - b.

        If the formula is not a predicate, we recursively traverse the
        subformulas associated with this formula, adding new subformula
        robustness scores rho_i for each subformula and defining

            rho = min_i{ rho_i }

        if the subformulas are combined with conjunction and

            rho = max_i{ rho_i }

        if the subformulas are combined with disjuction.
        """
        # We're at the bottom of the tree, so add the predicate constraints
        if isinstance(formula, STLPredicate):
            # rho = a'y - b
            y = self.y[:,t]
            self.mp.AddConstraint(eq( formula.a.T@y - formula.b, rho ))
        
        # We haven't reached the bottom of the tree, so keep adding
        # boolean constraints recursively
        else:
            rho_subs = []
            for i, subformula in enumerate(formula.subformula_list):
                rho_sub = self.mp.NewContinuousVariables(1)
                t_sub = formula.timesteps[i]   # the timestep at which this formula 
                                               # should hold
                self.AddSubformulaConstraints(subformula, rho_sub, t+t_sub)
                rho_subs.append(rho_sub)

            if formula.combination_type == "and":
                # rho = min(rho_subs)
                self._add_min_constraint(rho, rho_subs)

            else:  # combination_type == "or":
                # rho = max(rho_subs)
                self._add_max_constraint(rho, rho_subs)

    def _add_max_constraint(self, a, b_lst):
        """
        Add constraints to the optimization problem such that

            a = max(b_1,b_2,...,b_N)

        where b_lst = [b_1,b_2,...,b_N]
        """
        if len(b_lst) == 2:
            self._encode_max(a, b_lst[0], b_lst[1])
        elif len(b_lst) == 1:
            self.mp.AddConstraint(eq( a, b_lst[0] ))
        else:
            c = self.mp.NewContinuousVariables(1)
            self._add_max_constraint(c, b_lst[1:])
            self._encode_max(a, b_lst[0], c)
    
    def _add_min_constraint(self, a, b_lst):
        """
        Add constraints to the optimization problem such that

            a = min(b_1,b_2,...,b_N)

        where b_lst = [b_1,b_2,...,b_N]
        """
        if len(b_lst) == 2:
            self._encode_min(a, b_lst[0], b_lst[1])
        elif len(b_lst) == 1:
            self.mp.AddConstraint(eq(a, b_lst[0]))
        else:
            c = self.mp.NewContinuousVariables(1)
            self._add_min_constraint(c, b_lst[1:])
            self._encode_min(a, b_lst[0], c)
    
    def _add_absolute_value_constraint(self, x, y):
        """
        Add an absolute vlaue constraint 

            y = |x|

        to the optimization problem using the complementarity constraints

            x = x+ - x-
            y = x+ + x-
            x+ >= 0
            x- >= 0
            x+x- = 0
        """
        x_plus = self.mp.NewContinuousVariables(1)
        x_minus = self.mp.NewContinuousVariables(1)

        self.mp.AddConstraint(eq(x, x_plus - x_minus))
        self.mp.AddConstraint(eq(y, x_plus + x_minus))

        # LCP constraint as nonconvex quadratic constraint
        self.mp.AddConstraint(ge(x_plus, 0.0))
        self.mp.AddConstraint(ge(x_minus, 0.0))
        self.mp.AddConstraint(x_plus.T@x_minus == 0.0)  # could relax here
        
    def _encode_max(self, a, b, c):
        """
        This very important method takes three Gurobi decision variables 
        (a,b,c) and adds constraints on the optimization problem such
        that

            a = max(b, c).

        using the fact that

            max(b,c) = 1/2(b + c) + 1/2|b-c|

        and that absolute value can be encoded with an LCP constraint.
        """
        abs_b_minus_c = self.mp.NewContinuousVariables(1)
        self.mp.AddConstraint(eq( a , 0.5*(b + c) + 0.5*abs_b_minus_c ))
        self._add_absolute_value_constraint(b-c, abs_b_minus_c)
    
    def _encode_min(self, a, b, c):
        """
        This very important method takes three Gurobi decision variables 
        (a,b,c) and adds constraints on the optimization problem such
        that

            a = min(b, c).

        using the fact that
            
            min(b,c) = 1/2(b + c) - 1/2|b-c|

        and that absolute value can be encoded with an LCP constraint.
        """
        abs_b_minus_c = self.mp.NewContinuousVariables(1)
        self.mp.AddConstraint(eq( a , 0.5*(b + c) - 0.5*abs_b_minus_c ))
        self._add_absolute_value_constraint(b-c, abs_b_minus_c)
