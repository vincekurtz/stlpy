from solvers.solver_base import STLSolver
import numpy as np
from pydrake.all import MathematicalProgram, GurobiSolver, MosekSolver, eq

class MICPSolver(STLSolver):
    """
    Given an STLFormula (spec) and a system of the form 

        x_{t+1} = A*x_t + B*u_t,
        y_t = [x_t;u_t],

    use the standard Mixed-Integer Convex Programming (MICP) approach to
    find a satisfying trajectory (x,u), i.e., solve the MICP

        min  x'Qx + u'Ru 
        s.t. x_{t+1} = A*x_t + B*u_t
             x0 fixed
             (x,u) satisfies (spec)

    where the final constraint is enforced by the addition of
    integer variables for each subformula. 

    For further details on this approach, see

        Raman V, et al. 
        Model predictive control with signal temporal logic specifications. 
        IEEE Conference on Decision and Control, 2014

        Sadraddini S, Belta C. 
        Formal synthesis of control strategies for positive monotone systems. 
        IEEE Trans. Autom. Control. 2019
    """

    def __init__(self, spec, A, B, Q, R, x0, T, M):
        """
        Initialize the solver.

        @param spec     An STLFormula describing the specification
        @param A        An (n,n) numpy matrix describing state dynamics
        @param B        A (n,m) numpy matrix describing control input dynamics
        @param Q        A (n,n) matrix specifing a running state cost
        @param R        A (m,m) matrix specifing a running control cost
        @param x0       The initial state of the system.
        @param T        An integer specifiying the number of timesteps.
        @param M        A large positive scalar, used for the "big-M" method
        """
        assert M > 0, "M should be a (large) positive scalar"
        super().__init__(spec, A, B, Q, R, x0, T)

        # Create the drake MathematicalProgram instance that will allow
        # us to interface with a MIP solver like Gurobi or Mosek
        self.mp = MathematicalProgram()

        # Create optimization variables for x and u
        self.x = self.mp.NewContinuousVariables(self.n, self.T, 'x')
        self.u = self.mp.NewContinuousVariables(self.m, self.T, 'u')

        # Add cost and constraints to the optimization problem
        self.AddRunningCost()
        self.AddDynamicsConstraints()
        self.AddSTLConstraints()

    def Solve(self):
        """
        Solve the optimization problem and return the optimal values of (x,u).
        """
        solver = GurobiSolver()
        #solver = MosekSolver()
        res = solver.Solve(self.mp)

        solve_time = res.get_solver_details().optimizer_time
        print("Solve time: ", solve_time)

        if res.is_success():
            x = res.GetSolution(self.x)
            u = res.GetSolution(self.u)

            y = np.vstack([x,u])
            rho = self.spec.robustness(y,0)
            print("Optimal robustness: ", rho[0])
        else:
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

    def AddRunningCost(self):
        """
        Add the running cost

            min x'Qx + u'Ru

        to the optimization problem. 
        """
        for t in range(self.T):
            self.mp.AddCost( self.x[:,t].T@self.Q@self.x[:,t] + self.u[:,t].T@self.R@self.u[:,t] )

    def AddSTLConstraints(self):
        """
        Add the STL constraints

            (x,u) |= specification

        to the optimization problem, via the recursive introduction
        of binary variables for all subformulas in the specification.
        """
        pass



