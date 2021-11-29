from solvers.base import STLSolver
from STL import STLPredicate
import numpy as np

import gurobipy as gp
from gurobipy import GRB

import time

class GurobiMICPSolver(STLSolver):
    """
    Given an :class:`.STLFormula` :math:`\\varphi` and a :class:`.LinearSystem`, 
    solve the optimization problem

    .. math:: 

        \max & \\rho^{\\varphi}(y_0,y_1,\dots,y_T)

        \\text{s.t. } & x_0 \\text{ fixed}

        & x_{t+1} = f(x_t, u_t) 

        & y_{t} = g(x_t, u_t)

        & \\rho^{\\varphi}(y_0,y_1,\dots,y_T) \geq 0

    with Gurobi using mixed-integer convex programming. This gives a globally optimal
    solution, but may be computationally expensive for long and complex specifications.
   
    .. note::

        This class implements a slight variation of the method described in

        Raman V, et al. 
        *Model predictive control with signal temporal logic specifications*. 
        IEEE Conference on Decision and Control, 2014

        where we enforce constraint satisfaction using subformula robustness 
        values :math:`\\rho^{\\varphi_i}` directly rather using the binary variables
        :math:`z^{\\varphi_i}`.


    :param spec:            An :class:`.STLFormula` describing the specification.
    :param sys:             A :class:`.LinearSystem` describing the system dynamics.
    :param x0:              A ``(n,1)`` numpy matrix describing the initial state.
    :param T:               A positive integer fixing the total number of timesteps :math:`T`.
    :param M:               (optional) A large positive scalar used to rewrite ``min`` and ``max`` as
                            mixed-integer constraints. Default is ``1000``.
    :param robustness_cost: (optional) Boolean flag for adding a linear cost to maximize
                            the robustness measure. Default is ``True``.
    """

    def __init__(self, spec, sys, x0, T, M=1000, robustness_cost=True):
        assert M > 0, "M should be a (large) positive scalar"
        super().__init__(spec, sys, x0, T)
        self.M = float(M)

        # Set up the optimization problem
        self.model = gp.Model("STL_MICP")

        # Set timeout (in seconds)
        self.model.setParam('TimeLimit', 10*60)
        
        print("Setting up optimization problem...")
        st = time.time()  # for computing setup time

        # Create optimization variables
        self.y = self.model.addMVar((self.sys.p, self.T), lb=-float('inf'), name='y')
        self.x = self.model.addMVar((self.sys.n, self.T), lb=-float('inf'), name='x')
        self.u = self.model.addMVar((self.sys.m, self.T), lb=-float('inf'), name='u')
        self.rho = self.model.addMVar(1,name="rho",lb=0.0) # lb sets minimum robustness

        # Add cost and constraints to the optimization problem
        self.AddDynamicsConstraints()
        self.AddSTLConstraints()
        self.AddRobustnessConstraint()
        if robustness_cost:
            self.AddRobustnessCost()
        
        print(f"Setup complete in {time.time()-st} seconds.")

    def AddControlBounds(self, u_min, u_max):
        for t in range(self.T):
            self.model.addConstr( u_min <= self.u[:,t] )
            self.model.addConstr( self.u[:,t] <= u_max )
    
    def AddStateBounds(self, x_min, x_max):
        for t in range(self.T):
            self.model.addConstr( x_min <= self.x[:,t] )
            self.model.addConstr( self.x[:,t] <= x_max )
    
    def AddQuadraticCost(self, Q, R):
        print("Warning: objective reset to minimize quadratic cost")
        cost = self.x[:,0]@Q@self.x[:,0] + self.u[:,0]@R@self.u[:,0]
        for t in range(1,self.T):
            cost += self.x[:,t]@Q@self.x[:,t] + self.u[:,0]@R@self.u[:,0]
        self.model.setObjective(cost, GRB.MINIMIZE)

    def AddRobustnessConstraint(self, rho_min=0.0):
        self.model.addConstr( self.rho >= rho_min )
    
    def Solve(self, verbose=False, presolve=True):
        self.model.optimize()

        if self.model.status == GRB.OPTIMAL:
            print("\nOptimal Solution Found!\n")
            x = self.x.X
            u = self.u.X
            rho = self.rho.X[0]
            
            # Report optimal cost and robustness
            print("Solve time: ", self.model.Runtime)
            print("Optimal robustness: ", rho)
            print("")
        else:
            print(f"\nOptimization failed with status {self.model.status}.\n")
            x = None
            u = None
            rho = -np.inf

        return (x,u,rho,self.model.Runtime)

    def AddDynamicsConstraints(self):
        # Initial condition
        self.model.addConstr( self.x[:,0] == self.x0 )

        # Dynamics
        for t in range(self.T-1):
            self.model.addConstr( 
                    self.x[:,t+1] == self.sys.A@self.x[:,t] + self.sys.B@self.u[:,t] )

            self.model.addConstr( 
                    self.y[:,t] == self.sys.C@self.x[:,t] + self.sys.D@self.u[:,t] )
            
        self.model.addConstr( 
                self.y[:,self.T-1] == self.sys.C@self.x[:,self.T-1] + self.sys.D@self.u[:,self.T-1] )

    def AddRobustnessCost(self):
        print("Warning: objective reset to maximize robustness measure")
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

        If the formula is a predicate (a'y-b>=0), this means that

            rho = a'y(t) - b.

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
            # rho = a.T*[x;u] - b
            self.model.addConstr( formula.a.T@self.y[:,t] - formula.b == rho )
        
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
        This very important method converts a non-convex contraint

        .. math::

            a = \max(b, c)

        into a mixed-integer convex constraint using the "big-M" method:

        .. math::

            & a \geq b

            & a \geq c

            & a \leq b + Mz

            & a \leq c + M(1-z)

            & z \in \{0,1\}

        where :math:`M` is a large integer.
        """
        z = self.model.addMVar(1,vtype=GRB.BINARY)  
        self.model.addConstr( a >= b )
        self.model.addConstr( a >= c )
        self.model.addConstr( a <= b + self.M*z )
        self.model.addConstr( a <= c + self.M*(1-z) )
    
    def _encode_min(self, a, b, c):
        """
        Add constraints
        
        .. math::

            & a \leq b

            & a \leq c

        to the optimization problem to enforce

        .. math::

            a = \min(b, c)

        .. note::

            This approach is only valid if the specification is in positive
            normal form (PNF).

        """
        self.model.addConstr( a <= b )
        self.model.addConstr( a <= c )
