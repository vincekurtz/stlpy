from solvers.solver_base import STLSolver
from STL import STLPredicate
import numpy as np

from pydrake.all import MathematicalProgram, eq, le, ge
from pydrake.solvers.all import IpoptSolver, SnoptSolver
from pydrake.solvers.nlopt import NloptSolver

import time

class DrakeLCPSolver(STLSolver):
    """
    Given an STLFormula (spec) and a system of the form 

        x_{t+1} = A*x_t + B*u_t,
        y_t = [x_t;u_t],

    use a new Linear Complementarity Problem (LCP) approach
    to find a maximally robust satisfying trajectory (x,u), i.e.,

        max  rho
        s.t. x_{t+1} = A*x_t + B*u_t
             x0 fixed
             rho(x,u) is the STL robustness measure

    using the Drake interface, which allows us to try a variety
    of different solvers.

    This is nearly identical to the standard MICP encoding, but instead
    of encoding the min/max operators using mixed-integer constraints,
    we encode them using linear complementarity constraints. 
    """

    def __init__(self, spec, A, B, x0, T):
        """
        Initialize the solver.

        @param spec     An STLFormula describing the specification
        @param A        An (n,n) numpy matrix describing state dynamics
        @param B        A (n,m) numpy matrix describing control input dynamics
        @param x0       The initial state of the system.
        @param T        An integer specifiying the number of timesteps.
        """
        super().__init__(spec, A, B, x0, T)
        self.M = 1000

        # Set up the optimization problem
        self.mp = MathematicalProgram()

        # Choose a solver
        self.solver = SnoptSolver()

        print("Setting up optimization problem...")
        st = time.time()  # for computing setup time

        # Create optimization variables
        self.x = self.mp.NewContinuousVariables(self.n, self.T, 'x')
        self.u = self.mp.NewContinuousVariables(self.m, self.T, 'u')
        self.rho = self.mp.NewContinuousVariables(1,'rho')

        # Add cost and constraints to the optimization problem
        self.AddDynamicsConstraints()
        self.AddSTLConstraints()
        self.AddRobustnessCost()
        self.AddControlBoundConstraints()
        
        print(f"Setup complete in {time.time()-st} seconds.")

    def Solve(self):
        """
        Solve the optimization problem and return the optimal values of (x,u).
        """
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
                self.x[:,t+1], self.A@self.x[:,t] + self.B@self.u[:,t]
            ))

    def AddControlBoundConstraints(self, u_max=1.0):
        """
        Add the constraints

            -u_max <= u_t <= u_max

        to the optimization problem
        """
        u = self.u.flatten()
        N = len(u)
        lb = -u_max*np.ones(N)
        ub = u_max*np.ones(N)
        self.mp.AddLinearConstraint(A=np.eye(N), lb=lb, ub=ub, vars=u)

    def AddRobustnessCost(self):
        """
        Set the cost of the optimization problem to maximize
        the overall robustness measure.
        """
        self.mp.AddCost(-self.rho[0])

    def AddSTLConstraints(self):
        """
        Add the STL constraints

            (x,u) |= specification

        to the optimization problem, via the recursive introduction
        of binary variables for all subformulas in the specification.
        """
        # Constraint the overall formula robustness to be positive
        self.mp.AddConstraint(self.rho[0] >= 0)

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
        # We're at the bottom of the tree, so add the big-M constraints
        if isinstance(formula, STLPredicate):
            # rho = A[x;u] - b
            xu = np.hstack([self.x[:,t],self.u[:,t]])
            self.mp.AddConstraint(eq( formula.A@xu - formula.b, rho ))
        
        # We haven't reached the bottom of the tree, so keep adding
        # boolean constraints recursively
        else:
            if formula.combination_type == "and":
                rho_subs = []
                for i, subformula in enumerate(formula.subformula_list):
                    rho_sub = self.mp.NewContinuousVariables(1)
                    t_sub = formula.timesteps[i]   # the timestep at which this formula 
                                                   # should hold
                    self.AddSubformulaConstraints(subformula, rho_sub, t+t_sub)
                    rho_subs.append(rho_sub)
             
                # rho = min(rho_subs)
                self._add_min_constraint(rho, rho_subs)

            else:  # combination_type == "or":
                rho_subs = []
                for i, subformula in enumerate(formula.subformula_list):
                    rho_sub = self.mp.NewContinuousVariables(1)
                    t_sub = formula.timesteps[i]   # the timestep at which this formula 
                                                   # should hold
                    self.AddSubformulaConstraints(subformula, rho_sub, t+t_sub)
                    rho_subs.append(rho_sub)
               
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

        M = np.array([[0.,1.],[0.,0.]])
        q = np.array([0.,0.])
        x = np.hstack([x_plus,x_minus])
        self.mp.AddLinearComplementarityConstraint(M,q,x)

        #self.mp.AddConstraint(ge(x_plus, 0.0))
        #self.mp.AddConstraint(ge(x_minus, 0.0))
        #self.mp.AddConstraint(x_plus.T@x_minus <= 0.1)

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
