from solvers.base import STLSolver
from STL import STLPredicate
import numpy as np

from knitro import *

import time

class KnitroLCPSolver(STLSolver):
    """
    Given an :class:`.STLFormula` :math:`\\varphi` and a :class:`.LinearSystem`, 
    solve the optimization problem

    .. math:: 

        \max & \\rho^{\\varphi}(y_0,y_1,\dots,y_T)

        \\text{s.t. } & x_0 \\text{ fixed}

        & x_{t+1} = f(x_t, u_t) 

        & y_{t} = g(x_t, u_t)

        & \\rho^{\\varphi}(y_0,y_1,\dots,y_T) \geq 0

    where :math:`\\rho^{\\varphi}` is defined using linear complementarity constraints
    rather than mixed-integer constraints. The knitro solver is able to exploit the
    structure of these LCP constraints to find high-quality locally optimal solutions.

    :param spec:            An :class:`.STLFormula` describing the specification.
    :param sys:             A :class:`.LinearSystem` describing the system dynamics.
    :param x0:              A ``(n,1)`` numpy matrix describing the initial state.
    :param T:               A positive integer fixing the total number of timesteps :math:`T`.
    :param robustness_cost: (optional) Boolean flag for adding a linear cost to maximize
                            the robustness measure. Default is ``True``.
    """
    def __init__(self, spec, sys, x0, T, robustness_cost=True):
        super().__init__(spec, sys, x0, T)

        # Set up the optimization problem
        self.kc = KN_new()

        # Set some parameters
        KN_set_int_param(self.kc, "par_numthreads", 4)  # Enable parallelism
        KN_set_int_param(self.kc, "ms_enable", 1)       # Enable multistart
        KN_set_int_param(self.kc, "ms_terminate",       # Stop when a feasible
                KN_MSTERMINATE_FEASIBLE)                # solution is found
        KN_set_int_param(self.kc, "algorithm", 0)       # Choose algorithm automatically

        print("Setting up optimization problem...")
        st = time.time()  # for computing setup time

        # Create primary optimization variables
        self.x_idx = np.empty((self.sys.n, self.T), dtype=int)  # Knitro references variables by index rather
        self.u_idx = np.empty((self.sys.m, self.T), dtype=int)  # than some custom class
        self.y_idx = np.empty((self.sys.p, self.T), dtype=int)
        for i in range(self.sys.n):
            self.x_idx[i,:] = KN_add_vars(self.kc,self.T)
        for i in range(self.sys.m):
            self.u_idx[i,:] = KN_add_vars(self.kc,self.T)
        for i in range(self.sys.p):
            self.y_idx[i,:] = KN_add_vars(self.kc,self.T)
        self.rho_idx = KN_add_vars(self.kc, 1)[0]

        # Add cost and constraints to the optimization problem
        self.AddDynamicsConstraints()
        self.AddSTLConstraints()
        self.AddRobustnessConstraint()
        if robustness_cost:
            self.AddRobustnessCost()

        print(f"Setup complete in {time.time()-st} seconds.")

    def AddControlBounds(self, u_min, u_max):
        for t in range(self.T):
            for i in range(self.sys.m):
                KN_set_var_lobnds(self.kc, self.u_idx[i,t], u_min[i])
                KN_set_var_upbnds(self.kc, self.u_idx[i,t], u_max[i])
    
    def AddStateBounds(self, x_min, x_max):
        for t in range(1,self.T):         # putting contraint on initial state would
            for i in range(self.sys.n):   # reset the initial condition constraint
                KN_set_var_lobnds(self.kc, self.x_idx[i,t], x_min[i])
                KN_set_var_upbnds(self.kc, self.x_idx[i,t], x_max[i])
    
    def AddOutputBounds(self, y_min, y_max):
        for t in range(1,self.T):         # putting contraint on initial state would
            for i in range(self.sys.n):   # reset the initial condition constraint
                KN_set_var_lobnds(self.kc, self.y_idx[i,t], y_min[i])
                KN_set_var_upbnds(self.kc, self.y_idx[i,t], y_max[i])
    
    def AddQuadraticCost(self, Q, R):
        # Check that off-diagonal elements of Q and R are zero
        Q_bar = Q*(1-np.eye(Q.shape[0]))
        R_bar = R*(1-np.eye(R.shape[0]))
        if not np.all(Q_bar == 0) or not np.all(R_bar == 0):
            raise ValueError("Only diagonal Q/R are supported at this time")

        q = np.diag(Q)
        r = np.diag(R)
        for t in range(self.T):
            for i in range(self.sys.n):
                KN_add_obj_quadratic_struct(self.kc, self.x_idx[i,t], self.x_idx[i,t], q[i])
            for i in range(self.sys.m):
                KN_add_obj_quadratic_struct(self.kc, self.u_idx[i,t], self.u_idx[i,t], r[i])
    
    def AddRobustnessConstraint(self, rho_min=0.0):
        KN_set_var_lobnds(self.kc, self.rho_idx, rho_min)
    
    def Solve(self):
        st = time.time()
        status = KN_solve(self.kc)
        solve_time = time.time() - st

        if status > -200:  # feasible return codes, see https://www.artelys.com/docs/knitro/3_referenceManual/returnCodes.html
            print("\nFeasible Solution Found!\n")

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
            rho = self.spec.robustness(y,0)[0]
            print("Solve time: ", solve_time)
            print("Optimal robustness: ", rho)

        else:
            print(f"\nFailed with status code {status}\n")
            x = None
            u = None
            rho = -np.inf

        return (x,u,rho,solve_time)

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
        for t in range(self.T-1):
            # x_{t+1} = Ax_t + Bu_t
            #
            # Formulate as M*x = b, where
            # M = [-I, A, B], x = [x_{t+1},x_t,u_t], b = 0
            M = np.hstack([-np.eye(self.sys.n), self.sys.A, self.sys.B])
            x_idx = np.hstack([self.x_idx[:,t+1],self.x_idx[:,t],self.u_idx[:,t]])
            b = np.zeros(x_idx.shape)
            add_linear_eq_cons(self.kc, M, x_idx, b)

            # y_t = Cx_t + Du_t
            M = np.hstack([-np.eye(self.sys.p), self.sys.C, self.sys.D])
            x_idx = np.hstack([self.y_idx[:,t],self.x_idx[:,t],self.u_idx[:,t]])
            b = np.zeros(x_idx.shape)
            add_linear_eq_cons(self.kc, M, x_idx, b)
        
        # y_T = Cx_T + Du_T
        M = np.hstack([-np.eye(self.sys.p), self.sys.C, self.sys.D])
        x_idx = np.hstack([self.y_idx[:,self.T-1],self.x_idx[:,self.T-1],self.u_idx[:,self.T-1]])
        b = np.zeros(x_idx.shape)
        add_linear_eq_cons(self.kc, M, x_idx, b)

    def AddRobustnessCost(self):
        """
        Set the cost of the optimization problem to maximize
        the overall robustness measure.
        """
        KN_add_obj_linear_struct(self.kc, self.rho_idx, -1.0)

    def AddRunningCost(self, Q, R):
        """
        Add a running cost

            \sum_t x'Qx + u'Ru

        to the optimization problem. 
        """
        # Check that off-diagonal elements of Q and R are zero
        Q_bar = Q*(1-np.eye(Q.shape[0]))
        R_bar = R*(1-np.eye(R.shape[0]))
        if not np.all(Q_bar == 0) or not np.all(R_bar == 0):
            raise ValueError("Only diagonal Q/R are supported at this time")

        q = np.diag(Q)
        r = np.diag(R)
        for t in range(self.T):
            for i in range(self.n):
                KN_add_obj_quadratic_struct(self.kc, self.x_idx[i,t], self.x_idx[i,t], q[i])
            for i in range(self.m):
                KN_add_obj_quadratic_struct(self.kc, self.u_idx[i,t], self.u_idx[i,t], r[i])

    def AddSTLConstraints(self):
        """
        Add the STL constraints

            (x,u) |= specification

        to the optimization problem, via the recursive introduction
        of binary variables for all subformulas in the specification.
        """
        # Set up lists of variable indices that make up all the complementarity
        # constraints. This is necessary because Knitro (for some reason) only
        # allows adding ALL the complementarity contraints once. 
        self.comp_cons = ([],[])

        # Recursively traverse the tree defined by the specification
        # to add constraints that define the STL robustness score
        self.AddSubformulaConstraints(self.spec, self.rho_idx, 0)

        # Add the complementarity constraints to the optimization problem
        cc_type = [KN_CCTYPE_VARVAR]*len(self.comp_cons[0])
        KN_set_compcons(self.kc, cc_type, self.comp_cons[0], self.comp_cons[1])

    def AddSubformulaConstraints(self, formula, rho_idx, t):
        """
        Given an STLFormula (formula) and a continuous variable idex (rho_idx),
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
        #   rho = A[x;u] - b
        if isinstance(formula, STLPredicate):
            # reformulate as M*x = b
            x_idx = np.hstack([self.y_idx[:,t],rho_idx])
            M = np.hstack([formula.a.T, -np.eye(1)])
            b = formula.b
            add_linear_eq_cons(self.kc, M, x_idx, b)
        
        # We haven't reached the bottom of the tree, so keep adding
        # boolean constraints recursively
        else:
            rho_subs = []
            for i, subformula in enumerate(formula.subformula_list):
                rho_sub_idx = KN_add_vars(self.kc, 1)[0]
                t_sub = formula.timesteps[i]   # the timestep at which this formula 
                                               # should hold
                self.AddSubformulaConstraints(subformula, rho_sub_idx, t+t_sub)
                rho_subs.append(rho_sub_idx)
             
            if formula.combination_type == "and":
                # rho = min(rho_subs)
                self._add_min_constraint(rho_idx, rho_subs)

            else:  # combination_type == "or":
                # rho = max(rho_subs)
                self._add_max_constraint(rho_idx, rho_subs)

    def _add_max_constraint(self, a, b_lst):
        """
        Add constraints to the optimization problem such that

            a = max(b_1,b_2,...,b_N)

        where b_lst = [b_1,b_2,...,b_N]
        """
        if len(b_lst) == 2:
            self._encode_max(a, b_lst[0], b_lst[1])
        elif len(b_lst) == 1:
            # a = b_lst[0]
            add_linear_eq_cons(self.kc, np.array([[1,-1]]),np.array([a,b_lst[0]]), np.array([0]))
        else:
            c = KN_add_vars(self.kc, 1)
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
            # a = b_lst[0]
            add_linear_eq_cons(self.kc, np.array([[1,-1]]),np.array([a,b_lst[0]]), np.array([0]))
        else:
            c = KN_add_vars(self.kc, 1)
            self._add_min_constraint(c, b_lst[1:])
            self._encode_min(a, b_lst[0], c)
    
    def _add_absolute_value_constraint(self, x_idx, y_idx):
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
        # Define x+ and x-
        x_plus_idx = KN_add_vars(self.kc, 1)
        x_minus_idx = KN_add_vars(self.kc, 1)
        KN_set_var_lobnds(self.kc, [x_plus_idx[0], x_minus_idx[0]], [0,0])

        # Constraint x = x+ - x-, y = x+ + x-
        Aeq = np.array([[-1,0,1,-1],[0,-1,1,1]])
        xeq_idx = np.hstack([x_idx, y_idx, x_plus_idx, x_minus_idx])
        beq = np.zeros(2)
        add_linear_eq_cons(self.kc, Aeq, xeq_idx, beq)

        # Add x+ and x- to the list of complementary variables so that a single
        # complementarity constraint with all the variables can be added later.
        self.comp_cons[0].append(x_plus_idx[0])
        self.comp_cons[1].append(x_minus_idx[0])

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
        # Create new variable for |b-c|
        abs_idx = KN_add_vars(self.kc,1)

        # Create new variable for b-c
        diff_idx = KN_add_vars(self.kc,1)

        # Add a linear constraint 1/2(b+c) + 1/2|b-c| = a
        Aeq = np.array([[-1,0.5,0.5,0.5]])
        x_idx = np.hstack([a, b, c, abs_idx])
        beq = np.zeros(1)
        add_linear_eq_cons(self.kc, Aeq, x_idx, beq)

        # Add a linear constraint diff = b-c
        Aeq = np.array([[1,-1,-1]])
        x_idx = np.hstack([b,c,diff_idx])
        beq = np.zeros(1)
        add_linear_eq_cons(self.kc, Aeq, x_idx, beq)

        self._add_absolute_value_constraint(diff_idx, abs_idx)
    
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
        # Create new variable for |b-c|
        abs_idx = KN_add_vars(self.kc,1)

        # Create new variable for b-c
        diff_idx = KN_add_vars(self.kc,1)

        # Add a linear constraint 1/2(b+c) - 1/2|b-c| = a
        Aeq = np.array([[-1,0.5,0.5,-0.5]])
        x_idx = np.hstack([a, b, c, abs_idx])
        beq = np.zeros(1)
        add_linear_eq_cons(self.kc, Aeq, x_idx, beq)

        # Add a linear constraint diff = b-c
        Aeq = np.array([[1,-1,-1]])
        x_idx = np.hstack([b,c,diff_idx])
        beq = np.zeros(1)
        add_linear_eq_cons(self.kc, Aeq, x_idx, beq)

        self._add_absolute_value_constraint(diff_idx, abs_idx)

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
            
