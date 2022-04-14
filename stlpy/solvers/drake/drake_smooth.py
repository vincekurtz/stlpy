from .drake_base import DrakeSTLSolver
from ...STL import LinearPredicate, NonlinearPredicate
import numpy as np

from pydrake.all import eq
from pydrake.solvers.all import IpoptSolver, SnoptSolver, SolverOptions, CommonSolverOption

import time

class DrakeSmoothSolver(DrakeSTLSolver):
    """
    Given an :class:`.STLFormula` :math:`\\varphi` and a :class:`.LinearSystem`,
    solve the optimization problem

    .. math::

        \max ~& \\rho^{\\varphi}(y_0,y_1,\dots,y_T)

        \\text{s.t. } & x_0 \\text{ fixed}

        & x_{t+1} = f(x_t, u_t)

        & y_{t} = g(x_t, u_t)

    where :math:`\\rho^{\\varphi}` is defined using a smooth approximation
    of min and max. We then use one of the general nonlinear solvers availible
    in Drake to find a locally optimal solution.

    .. Note::

        This class implements the algorithm described in

        Gilpin, Y, et al. *A Smooth Robustness Measure of Signal Temporal Logic for Symbolic Control*.
        IEEE Control Systems Letters, 2021. https://arxiv.org/abs/2006.05239.

    .. Note::

        This method is most effective when used in conjunction with the SNOPT
        sparse SQP solver. SNOPT is included with the binary version of Drake, but
        requires a license when built from source. 

    :param spec:    An :class:`.STLFormula` describing the specification.
    :param sys:     A :class:`.LinearSystem` describing the system dynamics.
    :param x0:      A ``(n,1)`` numpy matrix describing the initial state.
    :param T:       A positive integer fixing the total number of timesteps :math:`T`.
    :param k:       (optional). A smoothing parameter characterizing the tightness of
                    the smooth approximation. Larger values give a tighter approximation.
    :param verbose: (optional) A boolean indicating whether to print detailed
                    solver info. Default is ``True``.
    """

    def __init__(self, spec, sys, x0, T, k=2.0, verbose=True):
        DrakeSTLSolver.__init__(self, spec, sys, x0, T, verbose)
        self.k = k
        
        if self.verbose:
            print("Setting up optimization problem...")
            st = time.time()  # for computing setup time

        # Choose a solver
        snopt_solver = SnoptSolver()
        if snopt_solver.available():
            self.solver = snopt_solver
        else:
            print("\nWarning: this Drake installation does not include the sparse SQP solver SNOPT.")
            print("We'll try to use IPOPT instead, but this is slower and less reliable.")
            print("See https://drake.mit.edu/doxygen_cxx/classdrake_1_1solvers_1_1_snopt_solver.html for more details.\n")
            self.solver = IpoptSolver()

        # Add cost and constraints to the optimization problem
        self.AddDynamicsConstraints()
        self.AddSTLConstraints()
        self.AddRobustnessCost()

        if self.verbose:
            print(f"Setup complete in {time.time()-st} seconds.")

    def AddDynamicsConstraints(self):
        # Initial condition
        self.mp.AddConstraint(eq( self.x[:,0], self.x0 ))

        # Dynamics
        for t in range(self.T-1):
            self.mp.AddConstraint(eq(
                self.x[:,t+1], self.sys.f(self.x[:,t], self.u[:,t])
            ))
            self.mp.AddConstraint(eq(
                self.y[:,t], self.sys.g(self.x[:,t], self.u[:,t])
            ))
        self.mp.AddConstraint(eq(
            self.y[:,self.T-1], self.sys.g(self.x[:,self.T-1], self.u[:,self.T-1])
        ))

    def Solve(self):

        # Set solver options
        options = SolverOptions()
        if self.verbose:
            options.SetOption(CommonSolverOption.kPrintToConsole,1)
        self.mp.SetSolverOptions(options)

        # Local solvers tend to be sensitive to the initial guess
        np.random.seed(0)
        initial_guess = np.random.normal(size=self.mp.initial_guess().shape)

        st = time.time()
        res = self.solver.Solve(self.mp, initial_guess=initial_guess)
        solve_time = time.time() - st

        if res.is_success():
            if self.verbose:
                print("\nOptimal Solution Found!\n")
            x = res.GetSolution(self.x)
            u = res.GetSolution(self.u)

            # Report solve time and robustness
            y = self.sys.g(x, u)
            rho = self.spec.robustness(y,0)[0]
            if self.verbose:
                print("Solve time: ", solve_time)
                print("Optimal robustness: ", rho)

        else:
            if self.verbose:
                print("\nNo solution found.\n")
            x = None
            u = None
            rho = -np.inf

        return (x,u,rho,solve_time)

    def AddSTLConstraints(self):
        """
        Add the STL constraints

            (x,u) |= specification

        to the optimization problem, via the recursive introduction
        of binary variables for all subformulas in the specification.
        """
        # Recursively traverse the tree defined by the specification
        # to add constraints that define the STL robustness score
        self.AddSubformulaConstraints(self.spec, np.array([self.rho]), 0)

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
        if isinstance(formula, LinearPredicate):
            # rho = a'y - b
            y = self.y[:,t]
            self.mp.AddConstraint(eq( formula.a.T@y - formula.b, rho ))
        
        elif isinstance(formula, NonlinearPredicate):
            # rho = g(y)
            y = self.y[:,t]
            self.mp.AddConstraint(eq( formula.g(y), rho ))

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

            a ~= max(b_1,b_2,...,b_N)

        where b_lst = [b_1,b_2,...,b_N] useing a smooth approximation of
        the max operator.
        """
        if len(b_lst) == 1:
            self.mp.AddConstraint(eq( a, b_lst[0] ))
        else:
            x = np.array(b_lst)
            exp = np.exp(self.k*x) + 1e-12   # avoid divide by zero error

            self.mp.AddConstraint(eq(
                a , np.sum(x*exp)/np.sum(exp)
            ))

    def _add_min_constraint(self, a, b_lst):
        """
        Add constraints to the optimization problem such that

            a ~= max(b_1,b_2,...,b_N)

        where b_lst = [b_1,b_2,...,b_N] useing a smooth approximation of
        the min operator.
        """
        if len(b_lst) == 1:
            self.mp.AddConstraint(eq( a, b_lst[0] ))
        else:
            x = np.hstack(b_lst)

            self.mp.AddConstraint(eq(
                a , -1./float(self.k) * np.log(np.sum(np.exp(-self.k*x)))
            ))

