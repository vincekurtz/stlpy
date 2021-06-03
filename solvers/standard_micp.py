from solvers.solver_base import STLSolver
from STL import STLPredicate
import numpy as np
from pydrake.all import (MathematicalProgram, 
                         GurobiSolver, 
                         MosekSolver, 
                         eq)

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

    def __init__(self, spec, A, B, Q, R, x0, T, M, relaxed=False):
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
        @param relaxed  (optional) A boolean indicating whether to solve
                        a convex relaxation of the problem. Default to False.
        """
        assert M > 0, "M should be a (large) positive scalar"
        super().__init__(spec, A, B, Q, R, x0, T)
        self.M = M

        # Create optimization variables for x and u
        self.x = self.mp.NewContinuousVariables(self.n, self.T, 'x')
        self.u = self.mp.NewContinuousVariables(self.m, self.T, 'u')

        # Flag for whether to use a convex relaxation
        self.convex_relaxation = relaxed

        # Add cost and constraints to the optimization problem
        self.AddRunningCost()
        self.AddDynamicsConstraints()
        self.AddSTLConstraints()

    def Solve(self, verbose=False):
        """
        Solve the optimization problem and return the optimal values of (x,u).
        """

        # Print out some solver data
        num_continuous_variables, num_binary_variables = self.GetVariableData()
        print("Solving MICP with")
        print("    %s binary variables" % num_binary_variables)
        print("    %s continuous variables" % num_continuous_variables)

        # Set up the solver and solve the optimization problem
        solver = GurobiSolver()
        #solver = MosekSolver()

        if verbose:
            self.mp.SetSolverOption(solver.solver_id(), "OutputFlag", 1)
            self.mp.SetSolverOption(solver.solver_id(), "Presolve", 2)

        res = solver.Solve(self.mp)

        solve_time = res.get_solver_details().optimizer_time
        print("Solve time: ", solve_time)

        if res.is_success():
            x = res.GetSolution(self.x)
            u = res.GetSolution(self.u)

            y = np.vstack([x,u])
            rho = self.spec.robustness(y,0)
            print("Optimal Cost: ", res.get_optimal_cost())
            print("Optimal robustness: ", rho[0])
        else:
            print("No solution found")
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
        # Add a binary variable which takes a value of 1 only 
        # if the overall specification is satisfied.
        z_spec = self.NewBinaryVariables(1)
        self.mp.AddConstraint(eq( z_spec, 1 ))

        # Recursively traverse the tree defined by the specification
        # subformulas and add similar binary constraints. 
        self.AddSubformulaConstraints(self.spec, z_spec, 0)

    def AddSubformulaConstraints(self, formula, z, t):
        """
        Given an STLFormula (formula) and a binary variable (z),
        add constraints to the optimization problem such that z
        takes value 1 only if the formula is satisfied (at time t). 

        If the formula is a predicate, this constraint uses the "big-M" 
        formulation

            A[x(t);u(t)] - b + (1-z)M >= 0,

        which enforces A[x;u] - b >= 0 if z=1, where (A,b) are the 
        linear constraints associated with this predicate. 

        If the formula is not a predicate, we recursively traverse the
        subformulas associated with this formula, adding new binary 
        variables z_i for each subformula and constraining

            z <= z_i  for all i

        if the subformulas are combined with conjunction (i.e. all 
        subformulas must hold), or otherwise constraining

            z <= sum(z_i)

        if the subformulas are combined with disjuction (at least one
        subformula must hold). 
        """
        # We're at the bottom of the tree, so add the big-M constraints
        if isinstance(formula, STLPredicate):
            # A[x;u] - b + (1-z)*M >= 0
            A = np.hstack([formula.A,-np.array([[self.M]])])
            lb = formula.b - self.M
            ub = np.array([np.inf])
            vars = np.hstack([self.x[:,t],self.u[:,t],z])
            self.mp.AddLinearConstraint(A=A, lb=lb, ub=ub, vars=vars)
        
        # We haven't reached the bottom of the tree, so keep adding
        # boolean constraints recursively
        else:
            if formula.combination_type == "and":
                for i, subformula in enumerate(formula.subformula_list):
                    z_sub = self.NewBinaryVariables(1)
                    t_sub = formula.timesteps[i]   # the timestep at which this formula 
                                                   # should hold
                    self.AddSubformulaConstraints(subformula, z_sub, t+t_sub)
                    self.mp.AddConstraint( z[0] <= z_sub[0] )

            else:  # combination_type == "or":
                z_subs = []
                for i, subformula in enumerate(formula.subformula_list):
                    z_sub = self.NewBinaryVariables(1)
                    t_sub = formula.timesteps[i]
                    z_subs.append(z_sub)
                    self.AddSubformulaConstraints(subformula, z_sub, t+t_sub)

                # z <= sum(z_subs)
                A = np.hstack([1,-np.ones(len(z_subs))])[np.newaxis]
                lb = -np.array([np.inf])
                ub = np.array([0])
                vars = np.vstack([z,z_subs])
                self.mp.AddLinearConstraint(A=A, lb=lb, ub=ub, vars=vars)
