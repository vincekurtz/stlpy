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
    def __init__(self, spec, sys, x0, T, solver='gurobi'):
        super().__init__(spec, sys, x0, T)

        # Choose which solver to use
        if solver == 'gurobi':
            self.solver = GurobiSolver()
        elif solver == 'mosek':
            self.solver = MosekSolver()
        else:
            print("Using Naive Branch-and-Bound solver")
            self.solver = "bnb"
        
        # Define global bounds on y
        # TODO: let the user determine this
        y_min = np.array([0,0,-1,-1,-0.5,-0.5])
        y_max = np.array([10,10,1,1,0.5,0.5])
        A0 = np.vstack([np.eye(6),-np.eye(6)])
        b0 = np.hstack([y_max,-y_min])

        # Get list of all conjunctive state formulas
        self.CSFs = self.spec.get_all_conjunctive_state_formulas()
        self.n_csf = len(self.CSFs)

        # Define a list of indexes for all possible combinations of CSFs
        s = [i for i in range(self.n_csf)]
        self.powerset_idx = list(chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))

        # Define a "mode" (i.e., convex constraint A*y<=b) associated with each possible
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

        # Prune infeasible modes
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

        # Define (binary) variables for each mode at each timestep
        self.num_modes = len(self.powerset)
        self.z = self.mp.NewContinuousVariables(self.num_modes,self.T,'z')
        self.mp.AddConstraint(ge( self.z.flatten(), 0 ))
        
        # We can only be in one mode at each timestep
        self.mp.AddConstraint(eq(
            np.sum(self.z,axis=0), 1
        ))

        # Make copies of x, u and y for each mode at each timestep
        # These are indexed by [t,i,:], where the last dimension is m, n, or p
        X = []; U = []; Y = []
        for t in range(self.T):
            X.append(self.mp.NewContinuousVariables(self.num_modes, self.sys.n))
            U.append(self.mp.NewContinuousVariables(self.num_modes, self.sys.m))
            Y.append(self.mp.NewContinuousVariables(self.num_modes, self.sys.p))
        self.X = np.array(X)
        self.U = np.array(U)
        self.Y = np.array(Y)

        print("debug 1")

        # Define a binary variable a_t^{ij} for each possible transition between
        # modes at each timestep, and  store in a numpy array indexed by [t,i,j], 
        # where
        #   t is the starting time-index
        #   i is the starting mode
        #   j is the ending mode (i.e., the mode at t+1)
        self.a = np.empty((self.T-1, self.num_modes, self.num_modes), dtype=object)
        for t in range(self.T-1):
            self.a[t,:,:] = self.mp.NewBinaryVariables(self.num_modes, self.num_modes, 'a')

            # DEBUG: convex relaxation
            #self.a[t,:,:] = self.mp.NewContinuousVariables(self.num_modes, self.num_modes, 'a')
            #self.mp.AddConstraint(ge( self.a[t,:,:].flatten(), 0 ))

            # Must have exactly one active edge at each timestep
            summ = np.sum(self.a[t,:,:])
            self.mp.AddConstraint(summ == 1)

        # Relate binary flow through each mode to its occupancy index z_t^i
        for t in range(0,self.T-1):
            for i in range(self.num_modes):
                output_flow = np.sum(self.a[t,i,:])
                self.mp.AddConstraint( self.z[i,t] == output_flow )
        for t in range(1,self.T):
            for i in range(self.num_modes):
                input_flow = np.sum(self.a[t-1,:,i])
                self.mp.AddConstraint( self.z[i,t] == input_flow )
        
        print("debug 2")

        # Create 2 copies of state/control/output for each edge in the SPP graph, and
        # store in numpy arrays indexed by [t,i,j,:], where the last dimension
        # is n, m, or p
        self.X_start = np.empty(
              (self.T-1, self.num_modes, self.num_modes, self.sys.n), dtype=object)
        self.X_end = np.empty(
              (self.T-1, self.num_modes, self.num_modes, self.sys.n), dtype=object)
        self.U_start = np.empty(
                (self.T-1, self.num_modes, self.num_modes, self.sys.m), dtype=object)
        self.U_end = np.empty(
                (self.T-1, self.num_modes, self.num_modes, self.sys.m), dtype=object)
        self.Y_start = np.empty(
              (self.T-1, self.num_modes, self.num_modes, self.sys.p), dtype=object)
        self.Y_end = np.empty(
              (self.T-1, self.num_modes, self.num_modes, self.sys.p), dtype=object)
        for i in range(self.num_modes):
            for j in range(self.num_modes):
                self.X_start[:,i,j,:] = self.mp.NewContinuousVariables(self.T-1, self.sys.n)
                self.X_end[:,i,j,:] = self.mp.NewContinuousVariables(self.T-1, self.sys.n)
                self.U_start[:,i,j,:] = self.mp.NewContinuousVariables(self.T-1, self.sys.m)
                self.U_end[:,i,j,:] = self.mp.NewContinuousVariables(self.T-1, self.sys.m)
                self.Y_start[:,i,j,:] = self.mp.NewContinuousVariables(self.T-1, self.sys.p)
                self.Y_end[:,i,j,:] = self.mp.NewContinuousVariables(self.T-1, self.sys.p)
        
        print("debug 3")

        ## Recover original state, control, and output from these copies
        #x_start = np.sum(self.X_start, axis=(1,2))
        #x_end = np.sum(self.X_end, axis=(1,2))
        #self.mp.AddConstraint(eq(
        #    x_start, self.x[:,:self.T-1].T
        #))
        #self.mp.AddConstraint(eq(
        #    x_end, self.x[:,1:].T
        #))

        #y_start = np.sum(self.Y_start, axis=(1,2))
        #y_end = np.sum(self.Y_end, axis=(1,2))
        #self.mp.AddConstraint(eq(
        #    y_start, self.y[:,:self.T-1].T
        #))
        #self.mp.AddConstraint(eq(
        #    y_end, self.y[:,1:].T
        #))

        #u_start = np.sum(self.U_start, axis=(1,2))
        #u_end = np.sum(self.U_end, axis=(1,2))
        #self.mp.AddConstraint(eq(
        #    u_start, self.u[:,:self.T-1].T
        #))
        #self.mp.AddConstraint(eq(
        #    u_end, self.u[:,1:].T
        #))
        
        print("debug 4")

        # Recover state, control, and output for each mode from these copies
        for t in range(self.T-1):
            for i in range(self.num_modes):
                state_outflow = np.sum(self.X_start[t,i,:,:], axis=0)
                control_outflow = np.sum(self.U_start[t,i,:,:], axis=0)
                output_outflow = np.sum(self.Y_start[t,i,:,:], axis=0)
                self.mp.AddConstraint(eq(
                    self.X[t,i], state_outflow
                ))
                self.mp.AddConstraint(eq(
                    self.U[t,i], control_outflow
                ))
                self.mp.AddConstraint(eq(
                    self.Y[t,i], output_outflow
                ))
        for t in range(1,self.T):
            for i in range(self.num_modes):
                state_inflow = np.sum(self.X_end[t-1,:,i,:], axis=0)
                control_inflow = np.sum(self.U_end[t-1,:,i,:], axis=0)
                output_inflow = np.sum(self.Y_end[t-1,:,i,:], axis=0)
                self.mp.AddConstraint(eq(
                    self.X[t,i], state_inflow
                ))
                self.mp.AddConstraint(eq(
                    self.U[t,i], control_inflow
                ))
                self.mp.AddConstraint(eq(
                    self.Y[t,i], output_inflow
                ))

        # Add cost and constraints to the optimization problem
        print("debug 5")
        self.AddDynamicsConstraints()
        print("debug 6")
        self.AddSTLConstraints()
    
    def AddQuadraticCost(self, Q, R):
        """
        Add the running cost using the perspective formulation, which
        encodes a quadratic cost as a linear cost plus second-order cone
        constraints.
        """
        # Define slack variables
        l = self.mp.NewContinuousVariables(self.num_modes, self.T)
        self.mp.AddLinearConstraint(ge( l.flatten(), 0 ))
        self.mp.AddLinearCost(np.sum(l))

        for t in range(self.T-1):
            for i in range(self.num_modes):
                for j in range(self.num_modes):
                    x = self.X_start[t,i,j,:]
                    u = self.U_start[t,i,j,:]
                    a = self.a[t,i,j]

                    quad_expr = x.T@Q@x + u.T@R@u

                    self.mp.AddRotatedLorentzConeConstraint(l[i,t], a, quad_expr)

    def AddDynamicsConstraints(self):
        """
        Add the constraints

            x_{t+1} = A@x_t + B@u_t
            x_0 = x0

        to the optimization problem. 
        """
        # Copies of x, u, y sum to the original thing
        self.mp.AddConstraint(eq( self.x.T, np.sum(self.X,axis=1) ))
        self.mp.AddConstraint(eq( self.u.T, np.sum(self.U,axis=1) ))
        self.mp.AddConstraint(eq( self.y.T, np.sum(self.Y,axis=1) ))

        # Initial condition
        self.mp.AddConstraint(eq( self.x[:,0], self.x0 ))

        # Binary flow constraints
        #
        #   sum_i a_{t-1}^{ij} = sum_k a_t^{jk}
        for i in range(self.num_modes):
            for t in range(1,self.T-1):
                inflow = np.sum(self.a[t-1,:,i])
                outflow = np.sum(self.a[t,i,:])

                self.mp.AddConstraint( inflow == outflow )

        # Continuous flow constraints
        #
        #   sum_i x_end_{t-1}^{ij} = sum_k x_start_t^{jk}
        for i in range(self.num_modes):
            for t in range(1, self.T-1):
                inflow_x = np.sum(self.X_end[t-1,:,i,:], axis=0)
                outflow_x = np.sum(self.X_start[t,i,:,:],axis=0)
        
                inflow_y = np.sum(self.Y_end[t-1,:,i,:], axis=0)
                outflow_y = np.sum(self.Y_start[t,i,:,:],axis=0)
                
                inflow_u = np.sum(self.U_end[t-1,:,i,:], axis=0)
                outflow_u = np.sum(self.U_start[t,i,:,:],axis=0)
                
                self.mp.AddConstraint(eq( inflow_x, outflow_x ))
                self.mp.AddConstraint(eq( inflow_y, outflow_y ))
                self.mp.AddConstraint(eq( inflow_u, outflow_u ))


        # Dynamics constraints
        # x_end_t^{ij} = A x_start_t^{ij} + B u_start_t^{ij}
        for t in range(self.T-1):
            for i in range(self.num_modes):
                for j in range(self.num_modes):
                    x = self.X_start[t,i,j,:]
                    u = self.U_start[t,i,j,:]
                    x_next = self.X_end[t,i,j,:]

                    self.mp.AddConstraint(eq(
                        x_next, self.sys.A@x + self.sys.B@u
                    ))

        # Output constraints
        for i in range(self.num_modes):
            for j in range(self.num_modes):
                for t in range(self.T-1):
                    x = self.X_start[t,i,j,:]
                    u = self.U_start[t,i,j,:]
                    y = self.Y_start[t,i,j,:]
                    
                    self.mp.AddConstraint(eq(
                        y, self.sys.C@x + self.sys.D@u
                    ))
                for t in range(self.T-1):
                    x = self.X_end[t,i,j,:]
                    u = self.U_end[t,i,j,:]
                    y = self.Y_end[t,i,j,:]
                    
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
        # of binary variables a_ij
        for t in range(self.T-1):
            for i in range(self.num_modes):
                for j in range(self.num_modes):
                    y_start = self.Y_start[t,i,j,:]
                    y_end = self.Y_end[t,i,j,:]
                    a = self.a[t,i,j]

                    start_domain = self.powerset[i]
                    end_domain = self.powerset[j]
                    start_domain.AddPointInNonnegativeScalingConstraints(self.mp, y_start, a)
                    end_domain.AddPointInNonnegativeScalingConstraints(self.mp, y_end, a)

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

            # Get binary variables for all modes that include this formula
            zs = []
            for i in range(len(self.powerset_idx)):
                if idx in self.powerset_idx[i]:
                    zs.append(self.z[i,t])

            # At least one of these binary variables must be 1 if z=1
            # z = sum(z_i)  
            self.mp.AddConstraint(eq(z, sum(zs) ))  # == or <=

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
    
