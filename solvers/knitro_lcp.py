from solvers.solver_base import STLSolver
from STL import STLPredicate
import numpy as np

from knitro import *

import time

class KnitroLCPSolver(STLSolver):
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

    using the Artelys Knitro solver, which finds a locally optimal
    solution, but has some (seemingly secret) methods of handling
    complementarity constraints.

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
        self.kc = KN_new()

        print("Setting up optimization problem...")
        st = time.time()  # for computing setup time

        # Create primary optimization variables
        self.x_idx = np.empty((self.n, self.T), dtype=int)  # Knitro references variables by index rather
        self.u_idx = np.empty((self.m, self.T), dtype=int)  # than some custom class
        for i in range(self.n):
            self.x_idx[i,:] = KN_add_vars(self.kc,self.T)
        for i in range(self.m):
            self.u_idx[i,:] = KN_add_vars(self.kc,self.T)
        self.rho_idx = KN_add_vars(self.kc, 1)[0]

        # Add cost and constraints to the optimization problem
        self.AddDynamicsConstraints()
        #self.AddSTLConstraints()
        #self.AddRobustnessCost()
        
        print(f"Setup complete in {time.time()-st} seconds.")

    def Solve(self):
        """
        Solve the optimization problem and return the optimal values of (x,u).
        """

        st = time.time()
        status = KN_solve(self.kc)
        solve_time = time.time() - st

        if status == KN_RC_OPTIMAL_OR_SATISFACTORY:
            print("\nOptimal Solution Found!\n")

            _, obj, vals, _ = KN_get_solution(self.kc)

            # Extract the solution
            x = np.empty(self.x_idx.shape)
            u = np.empty(self.u_idx.shape)
            for i in range(self.x_idx.shape[0]):
                for j in range(self.x_idx.shape[1]):
                    idx = self.x_idx[i,j]
                    x[i,j] = vals[idx]
            for i in range(self.u_idx.shape[0]):
                for j in range(self.u_idx.shape[1]):
                    idx = self.u_idx[i,j]
                    u[i,j] = vals[idx]

            # Report solve time and robustness
            y = np.vstack([x,u])
            rho = self.spec.robustness(y,0)
            print("Solve time: ", solve_time)
            print("Optimal robustness: ", rho[0])

        else:
            print(f"\nFailed with status code {status}\n")
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
        #
        #   self.x[:,0] = self.x0
        #
        for i in range(len(self.x0)):
            idx = self.x_idx[i,0]
            val = self.x0[i]
            KN_set_var_fxbnds(self.kc, idx, val)

        # Dynamics
        #
        #   self.x[:,t+1] = self.A*self.x[:,t] + self.B*self.u[:,t]
        #
        for t in range(self.T-1):
            # Formulate as M*x = b, where
            # M = [-I, A, B], x = [x_{t+1},x_t,u_t], b = 0
            M = np.hstack([-np.eye(self.n), self.A, self.B])
            x_idx = np.hstack([self.x_idx[:,t+1],self.x_idx[:,t],self.u_idx[:,t]])
            b = np.zeros(x_idx.shape)

            add_linear_eq_cons(self.kc, M, x_idx, b)

    def AddRobustnessCost(self):
        """
        Set the cost of the optimization problem to maximize
        the overall robustness measure.
        """
        pass
        # TODO

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

        # Standard LCP constraint (SNOPT only)
        #M = np.array([[0.,1.],[0.,0.]])
        #q = np.array([0.,0.])
        #x = np.hstack([x_plus,x_minus])
        #self.mp.AddLinearComplementarityConstraint(M,q,x)
        
        # LCP constraint as nonconvex quadratic constraint
        #self.mp.AddConstraint(ge(x_plus, 0.0))
        #self.mp.AddConstraint(ge(x_minus, 0.0))
        #self.mp.AddConstraint(x_plus.T@x_minus <= 0.1)

        # LCP constraint as nonconvex quadratic cost
        #self.mp.AddConstraint(ge(x_plus, 0.0))
        #self.mp.AddConstraint(ge(x_minus, 0.0))
        #self.mp.AddCost(x_plus.T@x_minus)
        
        # LCP constraint as constraint on the Fischer function
        # where phi = 0 iff (0 <= x_plus) complements (x_minus >= 0)
        #phi = np.sqrt(x_plus**2 + x_minus**2) - x_plus - x_minus
        #self.mp.AddConstraint(eq( phi, 0 ))
        
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

def add_linear_eq_cons(kc, A, x_idx, b):
    """
    Helper function for adding the linear equality constriant

        A*x = b

    to the given knitro problem instance, where

        A \in (m,n)
        x \in (n)
        b \in (m)

    @param kc       Knitro problem instance
    @param A        Numpy array for the linear map
    @param x_idx    List or numpy array of Knitro indices of the decision variables
    @param b        Numpy array for the coefficient vector, shape 
    """
    m = A.shape[0]
    n = A.shape[1]

    # Add a (yet-to-be-defined) constraint for each row and get constraint index list
    con_idx = KN_add_cons(kc, m)

    for i in range(m):
        # Define the linear structure of a constraint for each row
        constraint_indices = [con_idx[i] for j in range(n)]
        variable_indices = x_idx
        coefficients = A[i,:]
        KN_add_con_linear_struct(kc, constraint_indices, variable_indices, coefficients)

        # Set the equality bounds of this constraint
        KN_set_con_eqbnds(kc, con_idx[i], cEqBnds=b[i])
            
