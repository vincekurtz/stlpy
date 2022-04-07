from ..base import STLSolver
from pydrake.all import MathematicalProgram, ge, le

class DrakeSTLSolver(STLSolver):
    """
    A base class for solvers that use the Drake interface to connect with
    a lower-level solver like Gurobi, Mosek, SNOPT, or IPOPT.
    """
    def __init__(self, spec, sys, x0, T, verbose):
        STLSolver.__init__(self, spec, sys, x0, T, verbose)

        # Create the drake MathematicalProgram instance that will allow
        # us to interface with a MIP solver like Gurobi or Mosek
        self.mp = MathematicalProgram()

        # Create optimization variables
        self.y = self.mp.NewContinuousVariables(self.sys.p, self.T, 'y')
        self.x = self.mp.NewContinuousVariables(self.sys.n, self.T, 'x')
        self.u = self.mp.NewContinuousVariables(self.sys.m, self.T, 'u')
        self.rho = self.mp.NewContinuousVariables(1,'rho')[0]

    def AddRobustnessConstraint(self, rho_min=0.0):
        self.mp.AddConstraint( self.rho >= rho_min )

    def AddRobustnessCost(self):
        self.mp.AddCost(-self.rho)

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
