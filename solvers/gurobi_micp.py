from solvers.solver_base import STLSolver
from STL import STLPredicate
import numpy as np

import gurobipy as gp
from gurobipy import GRB

import time

class GurobiMICPSolver(STLSolver):
    """
    Given an STLFormula (spec) and a system of the form 

        x_{t+1} = A*x_t + B*u_t,
        y_t = [x_t;u_t],

    use the standard Mixed-Integer Convex Programming (MICP) approach to
    find a maximally robust satisfying trajectory (x,u), i.e., solve the MICP

        max rho
        s.t. x_{t+1} = A*x_t + B*u_t
             x0 fixed
             rho(x,u) is the STL robustness measure

    using Gurobi's python bindings. 

    For further details on this approach, see

        Raman V, et al. 
        Model predictive control with signal temporal logic specifications. 
        IEEE Conference on Decision and Control, 2014

        Sadraddini S, Belta C. 
        Formal synthesis of control strategies for positive monotone systems. 
        IEEE Trans. Autom. Control. 2019
    """

    def __init__(self, spec, A, B, x0, T, M, relaxed=False):
        """
        Initialize the solver.

        @param spec     An STLFormula describing the specification
        @param A        An (n,n) numpy matrix describing state dynamics
        @param B        A (n,m) numpy matrix describing control input dynamics
        @param x0       The initial state of the system.
        @param T        An integer specifiying the number of timesteps.
        @param M        A large positive scalar, used for the "big-M" method
        @param relaxed  (optional) A boolean indicating whether to solve
                        a convex relaxation of the problem. Default to False.
        """
        assert M > 0, "M should be a (large) positive scalar"
        super().__init__(spec, A, B, x0, T)
        self.M = float(M)

        # Set up the optimization problem
        self.model = gp.Model("STL_MICP")
        
        print("Setting up optimization problem...")
        st = time.time()  # for computing setup time

        # Create optimization variables
        self.xu = self.model.addMVar((self.n+self.m, self.T), lb=-float('inf'), name='xu')
        self.x = self.xu[:self.n,:]
        self.u = self.xu[self.n:,:]
        self.rho = self.model.addMVar(1,name="rho",lb=0.0) # lb sets minimum robustness

        # Add cost and constraints to the optimization problem
        self.AddDynamicsConstraints()
        self.AddSTLConstraints()
        self.AddRobustnessCost()
        
        print(f"Setup complete in {time.time()-st} seconds.")

    def Solve(self, verbose=False, presolve=True):
        """
        Solve the optimization problem and return the optimal values of (x,u).
        """
        self.model.optimize()

        if self.model.status == GRB.OPTIMAL:
            print("\nOptimal Solution Found!\n")
            x = self.x.X
            u = self.u.X

            # Report optimal cost and robustness
            print("Solve time: ", self.model.Runtime)
            print("Optimal robustness: ", self.rho.X[0])
            print("")
        else:
            print(f"\nOptimization failed with status {self.model.status}.\n")
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
        self.model.addConstr( self.x[:,0] == self.x0 )

        # Dynamics
        for t in range(self.T-1):
            self.model.addConstr( 
                    self.x[:,t+1] == self.A@self.x[:,t] + self.B@self.u[:,t] )

    def AddRobustnessCost(self):
        """
        Set the cost of the optimization problem to maximize
        the overall robustness measure.
        """
        self.model.setObjective(1*self.rho, GRB.MAXIMIZE)

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
        # We're at the bottom of the tree, so add the big-M constraints
        if isinstance(formula, STLPredicate):
            # rho = A[x;u] - b
            self.model.addConstr( formula.A@self.xu[:,t] - formula.b == rho )
        
        # We haven't reached the bottom of the tree, so keep adding
        # boolean constraints recursively
        else:
            if formula.combination_type == "and":
                rho_subs = []
                for i, subformula in enumerate(formula.subformula_list):
                    rho_sub = self.model.addMVar(1, lb=-float('inf'))
                    t_sub = formula.timesteps[i]   # the timestep at which this formula 
                                                   # should hold
                    self.AddSubformulaConstraints(subformula, rho_sub, t+t_sub)
                    rho_subs.append(rho_sub)
             
                # rho = min(rho_subs)
                self._add_min_constraint(rho, rho_subs)

            else:  # combination_type == "or":
                rho_subs = []
                for i, subformula in enumerate(formula.subformula_list):
                    rho_sub = self.model.addMVar(1, lb=-float('inf'))
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
            self.model.addConstr(a == b_lst[0])
        else:
            c = self.model.addMVar(1,lb=-float('inf'))
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
            self.model.addConstr(a == b_lst[0])
        else:
            c = self.model.addMVar(1,lb=-float('inf'))
            self._add_min_constraint(c, b_lst[1:])
            self._encode_min(a, b_lst[0], c)
    
    def _encode_max(self, a, b, c):
        """
        This very important method takes three Gurobi decision variables 
        (a,b,c) and adds constraints on the optimization problem such
        that

            a = max(b, c).

        We do so by introducing a binary variable z and using the big-M
        method to enforce a=b if z=0 and a=c if z=1.
        """
        z = self.model.addMVar(1,vtype=GRB.BINARY)  
        self.model.addConstr( a >= b )
        self.model.addConstr( a >= c )
        self.model.addConstr( a <= b + self.M*z )
        self.model.addConstr( a <= c + self.M*(1-z) )
    
    def _encode_min(self, a, b, c):
        """
        This very important method takes three Gurobi decision variables 
        (a,b,c) and adds constraints on the optimization problem such
        that

            a = min(b, c).

        We do so by introducing a binary variable z and using the big-M
        method to enforce a=b if z=0 and a=c if z=1.
        """
        z = self.model.addMVar(1,vtype=GRB.BINARY)  
        self.model.addConstr( a <= b )
        self.model.addConstr( a <= c )
        self.model.addConstr( a >= b - self.M*z )
        self.model.addConstr( a >= c - self.M*(1-z) )
