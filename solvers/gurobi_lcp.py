from solvers.gurobi_micp import GurobiMICPSolver

import gurobipy as gp
from gurobipy import GRB

class GurobiLCPSolver(GurobiMICPSolver):
    """
    Given an STLFormula (spec) and a system of the form 

        x_{t+1} = A*x_t + B*u_t,
        y_t = [x_t;u_t],

    use a new Linear Complementarity Problem (LCP) approach to
    find a maximally robust satisfying trajectory (x,u), i.e.,

        max rho
        s.t. x_{t+1} = A*x_t + B*u_t
             x0 fixed
             rho(x,u) is the STL robustness measure

    using Gurobi's python bindings. 

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
        @param relaxed  (optional) A boolean indicating whether to solve
                        a convex relaxation of the problem. Default to False.
        """
        super().__init__(spec, A, B, x0, T, M=1000)  # M is just a placeholder here

        # Enable solving with nonconvex quadratic constraints
        self.model.params.NonConvex = 2

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
        x_plus = self.model.addMVar(1, lb=0.0)
        x_minus = self.model.addMVar(1, lb=0.0)

        self.model.addConstr(x == x_plus - x_minus)
        self.model.addConstr(y == x_plus + x_minus)
        self.model.addConstr(x_plus@x_minus == 0)

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
        abs_b_minus_c = self.model.addMVar(1)  # lb=0 is fine for absolute value
        self.model.addConstr( a == 0.5*(b + c) + 0.5*abs_b_minus_c )
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
        abs_b_minus_c = self.model.addMVar(1)  # lb=0 is fine for absolute value
        self.model.addConstr( a == 0.5*(b + c) - 0.5*abs_b_minus_c )
        self._add_absolute_value_constraint(b-c, abs_b_minus_c)
