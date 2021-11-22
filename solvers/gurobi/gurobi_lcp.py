from solvers.gurobi.gurobi_micp import GurobiMICPSolver

import gurobipy as gp
from gurobipy import GRB

class GurobiLCPSolver(GurobiMICPSolver):
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
    rather than mixed-integer constraints. Since this is a special case of a (nonconvex)
    Quadratically Constrainted Quadratic Program (QCQP), Gurobi is able to find a
    globally optimal solution, but this method may be computationally expensive for 
    long and complex specifications.
   
    .. note::

        This is nearly identical to the standard MICP encoding, but instead
        of encoding the min/max operators using mixed-integer constraints,
        we encode them using linear complementarity constraints. 

    :param spec:            An :class:`.STLFormula` describing the specification.
    :param sys:             A :class:`.LinearSystem` describing the system dynamics.
    :param x0:              A ``(n,1)`` numpy matrix describing the initial state.
    :param T:               A positive integer fixing the total number of timesteps :math:`T`.
    :param robustness_cost: (optional) Boolean flag for adding a linear cost to maximize
                            the robustness measure. Default is ``True``.
    """
    def __init__(self, spec, sys, x0, T, robustness_cost=True):
        super().__init__(spec, sys, x0, T, M=1, robustness_cost=robustness_cost)  # M is just a placeholder

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
        self.model.addConstr(x_plus@x_minus == 0.0)

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
