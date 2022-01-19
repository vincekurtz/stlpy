from solvers.drake.drake_base import DrakeSTLSolver
from STL import STLPredicate
import numpy as np
import time
from pydrake.all import (MathematicalProgram, HPolyhedron,
                         GurobiSolver, MosekSolver, ClpSolver,
                         SolverOptions, CommonSolverOption,
                         eq, le, ge)
from pydrake.solvers.branch_and_bound import MixedIntegerBranchAndBound

from itertools import chain, combinations

class DrakeTestSolver(DrakeSTLSolver):
    """
    Scratch solver for implementing research ideas
    """
    def __init__(self, spec, sys, x0, T, M=1000, relaxed=False, robustness_cost=True, solver='gurobi'):
        assert M > 0, "M should be a (large) positive scalar"
        super().__init__(spec, sys, x0, T)
        self.M = M

        # Choose which solver to use
        if solver == 'gurobi':
            self.solver = GurobiSolver()
        elif solver == 'mosek':
            self.solver = MosekSolver()
        else:
            print("Using Naive Branch-and-Bound solver")
            self.solver = "bnb"

        # Get list of all conjunctive state formulas
        self.CSFs = self.spec.get_all_conjunctive_state_formulas()
        self.n_csf = len(self.CSFs)

        # Define a list of indexes for all possible combinations of CSFs
        s = [i for i in range(self.n_csf)]
        self.powerset_idx = list(chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))

        # DEBUG: define global bounds on y
        # TODO: let the user determine this
        y_min = np.array([0,0,-1,-1,-0.5,-0.5])
        y_max = np.array([10,10,1,1,0.5,0.5])
        A0 = np.vstack([np.eye(6),-np.eye(6)])
        b0 = np.hstack([y_max,-y_min])

        # Define a list of inequalities (A*y<=b) associated with each possible
        # combination of CSFs.
        self.powerset = []
        for idx in self.powerset_idx:
            As = [A0]
            bs = [b0]
            for i in idx:
                state_formula = self.CSFs[i]
                A, b = state_formula.get_all_inequalities()
                As.append(A)
                bs.append(b)

            A = np.vstack(As)
            b = np.hstack(bs)

            poly = HPolyhedron(A,b)
            self.powerset.append(poly)

        # Prune infeasible combinations
        i = 0
        while i < len(self.powerset_idx):
            A = self.powerset[i].A()
            b = self.powerset[i].b()

            # solve a simple small LP to determine feasibility
            prog = MathematicalProgram()
            x = prog.NewContinuousVariables(A.shape[1])
            prog.AddLinearConstraint(le( A@x, b ))
            res = self.solver.Solve(prog)
            feasible = res.is_success()

            if feasible:
                # leave this set alone and go to the next one
                i += 1
            else:
                # Remove this set 
                self.powerset.pop(i)
                self.powerset_idx.pop(i)

        # Define binary variables for each element of the powerset at each timestep
        self.nz = len(self.powerset)
        self.z = self.mp.NewBinaryVariables(self.nz,self.T,'z')
        
        # DEBUG: continuous relaxation
        #self.z = self.mp.NewContinuousVariables(self.nz,self.T,'z')
        #self.mp.AddConstraint(ge( self.z.flatten(), 0 ))

        # Make copies of x, u and y for each element of the powerset at each timestep
        # These are indexed by [i,t,:], where the last dimension is m, n, or p
        X = []; U = []; Y = []
        for i in range(self.nz):
            X.append(self.mp.NewContinuousVariables(self.T, self.sys.n))
            U.append(self.mp.NewContinuousVariables(self.T, self.sys.m))
            Y.append(self.mp.NewContinuousVariables(self.T, self.sys.p))
        self.X = np.array(X)
        self.U = np.array(U)
        self.Y = np.array(Y)

        # We can only be in one element of the powerset at each timestep
        self.mp.AddConstraint(eq(
            np.sum(self.z,axis=0), 1
        ))

        # Flag for whether to use a convex relaxation
        self.convex_relaxation = relaxed

        # Add cost and constraints to the optimization problem
        self.AddDynamicsConstraints()
        self.AddSTLConstraints()
        self.AddRobustnessConstraint()
        if robustness_cost:
            self.AddRobustnessCost()
    
    def Solve(self):
        # Set verbose output
        options = SolverOptions()
        options.SetOption(CommonSolverOption.kPrintToConsole,1)
        #options.SetOption(GurobiSolver.id(), "Presolve", 2)
        self.mp.SetSolverOptions(options)
            
        if self.solver == "bnb":
            bnb_solver = MixedIntegerBranchAndBound(self.mp, ClpSolver.id())
            st = time.time()
            status = bnb_solver.Solve()
            solve_time = time.time() - st
            success = True
            res = bnb_solver

        else:
            res = self.solver.Solve(self.mp)
            success = res.is_success()
            solve_time = res.get_solver_details().optimizer_time
            
        print("")
        print("Solve time: ", solve_time)

        if success:
            x = res.GetSolution(self.x)
            u = res.GetSolution(self.u)

            y = np.vstack([x,u])
            rho = self.spec.robustness(y,0)[0]
            print("Optimal robustness: ", rho)
        else:
            print("No solution found")
            x = None
            u = None
            rho = -np.inf

        return (x,u, rho, solve_time)
    
    def AddQuadraticCost(self, Q, R):
        """
        Add the running cost using the perspective formulation, which
        encodes a quadratic cost as a linear cost plus second-order cone
        constraints.
        """
        # Define slack variables
        l = self.mp.NewContinuousVariables(self.nz, self.T)
        self.mp.AddLinearConstraint(ge( l.flatten(), 0 ))
        self.mp.AddLinearCost(np.sum(l))

        for t in range(self.T):
            for i in range(self.nz):
                x = self.X[i,t,:]
                u = self.U[i,t,:]
                z = self.z[i,t]

                quad_expr = x.T@Q@x + u.T@R@u

                self.mp.AddRotatedLorentzConeConstraint(l[i,t], z, quad_expr)

    def AddDynamicsConstraints(self):
        """
        Add the constraints

            x_{t+1} = A@x_t + B@u_t
            x_0 = x0

        to the optimization problem. 
        """
        # Copies of x, u, y sum to the original thing
        self.mp.AddConstraint(eq( self.x.T, np.sum(self.X,axis=0) ))
        self.mp.AddConstraint(eq( self.u.T, np.sum(self.U,axis=0) ))
        self.mp.AddConstraint(eq( self.y.T, np.sum(self.Y,axis=0) ))

        # Initial condition
        self.mp.AddConstraint(eq( self.x[:,0], self.x0 ))

        # Dynamics constraints
        for t in range(self.T-1):
            x_next = 0
            for i in range(self.nz):
                x = self.X[i,t,:]
                u = self.U[i,t,:]
                x_next += self.sys.A@x + self.sys.B@u

            self.mp.AddConstraint(eq(
                self.x[:,t+1], x_next
            ))

        # Output constraints
        for t in range(self.T):
            for i in range(self.nz):
                x = self.X[i,t,:]
                u = self.U[i,t,:]
                y = self.Y[i,t,:]

                self.mp.AddConstraint(eq(
                    y, self.sys.C@x + self.sys.D@u
                ))

    def AddSTLConstraints(self):
        """
        Add the STL constraints

            (x,u) |= specification

        to the optimization problem, via the recursive introduction
        of binary variables for all subformulas in the specification.
        """
        # Add constraints to enforce state formulas depending on the value
        # of binary variables z
        for t in range(self.T):
            for i in range(self.nz):
                y = self.Y[i,t,:]
                z = self.z[i,t]
                domain = self.powerset[i]
                domain.AddPointInNonnegativeScalingConstraints(self.mp, y, z)

        # Add a binary variable which takes a value of 1 only 
        # if the overall specification is satisfied.
        z_spec = self.mp.NewContinuousVariables(1)
        self.mp.AddConstraint(eq( z_spec, 1 ))

        # Recursively traverse the tree defined by the specification
        # subformulas and add similar binary constraints. 
        self.AddSubformulaConstraints(self.spec, z_spec, 0)

    def AddSubformulaConstraints(self, formula, z, t):
        """
        Given an STLFormula (formula) and a binary variable (z),
        add constraints to the optimization problem such that z
        takes value 1 only if the formula is satisfied (at time t). 

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
        # We're at the bottom of the tree
        if formula.is_conjunctive_state_formula():
            idx = self.CSFs.index(formula)

            # Get binary variables for all powersets that include this formula
            # Each z_i in zs enforces one of the possible combinations of conjunctive
            # state formulas that includes this CSF.
            zs = []
            for i in range(len(self.powerset_idx)):
                if idx in self.powerset_idx[i]:
                    zs.append(self.z[i,t])

            # z = sum(z_i)  
            self.mp.AddConstraint(eq(z, sum(zs) ))

        # We haven't reached the bottom of the tree, so keep adding
        # boolean constraints recursively
        else:
            z_subs = self.mp.NewContinuousVariables(len(formula.subformula_list),1)
            #z_subs = self.mp.NewBinaryVariables(len(formula.subformula_list),1)
            self.mp.AddConstraint(ge(z_subs, 0))
                    
            if formula.combination_type == "and":
                self.mp.AddConstraint(le( z, z_subs ))
            else:  # combination_type == "or":
                self.mp.AddConstraint(le( z, sum(z_subs) ))

            for i, subformula in enumerate(formula.subformula_list):
                t_sub = formula.timesteps[i]
                self.AddSubformulaConstraints(subformula, z_subs[i], t+t_sub)
