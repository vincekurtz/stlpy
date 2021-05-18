from solvers.solver_base import STLSolver
from STL import STLPredicate
from utils import *

import numpy as np
import scipy as sp
import time
import matplotlib.pyplot as plt
from pydrake.all import (MathematicalProgram, 
                         GurobiSolver, 
                         MosekSolver, 
                         eq)

class SPPMICPSolver(STLSolver):
    """
    Given an STLFormula (spec), a system of the form 

        x_{t+1} = A*x_t + B*u_t,
        y_t = [x_t;u_t],

    and bounds on x_t and u_t, use a new Mixed-Integer Convex Programming 
    (MICP) approach based Shortest Path Planning (SPP) through graphs of
    convex sets to find a satisfying trajectory. 
    """

    def __init__(self, spec, A, B, Q, R, x0, T):
        """
        Initialize the solver.

        @param spec     An STLFormula describing the specification
        @param A        An (n,n) numpy matrix describing state dynamics
        @param B        A (n,m) numpy matrix describing control input dynamics
        @param Q        A (n,n) matrix specifing a running state cost
        @param R        A (m,m) matrix specifing a running control cost
        @param x0       The initial state of the system.
        @param T        An integer specifiying the number of timesteps.
        """
        super().__init__(spec, A, B, Q, R, x0, T)

        # Create the drake MathematicalProgram instance that will allow
        # us to interface with a MIP solver like Gurobi or Mosek
        self.mp = MathematicalProgram()

        # Construct polytopic partitions
        self.partition_list = self.ConstructPartitions()

        # Construct a graph where each vertex corresponds to a partition at
        # a given timestep, and edges exist between each pair of partitions.
        V, E = self.ConstructGraph()
        nV = len(V)
        nE = len(E)

        # Add list of convex partitions corresponding to vertices
        P = [self.partition_list[s].polytope for (t,s) in V]

        # Determine whether to use a convex relaxation
        # TODO: set as optional argument, do similar for standard MICP
        convex_relaxation = True

        #############
        # DEBUG: set up a spp problem that gives us a minimum-cost path
        # to a target partition
        target = self.partition_list[7].polytope
        start = self.partition_list[4].polytope  # contains x0
        y0 = np.hstack([x0,np.zeros(self.m)])

        # Plot start and target regions
        #ax = plt.gca()
        #ax.set_xlim((0,10))
        #ax.set_ylim((0,10))
        #start.plot_2d(ax=ax)
        #target.plot_2d(ax=ax)
        #plt.show()

        # Get the vertices corresponding to the initial and final nodes
        # in the graph
        v0 = None
        vF = None
        for i, (t,s) in enumerate(V):
            if P[i] == start and t == 0:
                v0 = i
            if P[i] == target and t == self.T-1:
                vF = i

        # Add binary variables a_ij such that a_ij = 1 only if the edge
        # (i,j) is traversed in the optimal path
        if not convex_relaxation:
            a = self.mp.NewBinaryVariables(nE, 'a')
        else:
            a = self.mp.NewContinuousVariables(nE, 'a')
            for k in range(nE):
                self.mp.AddConstraint( 0 <= a[k] )
                self.mp.AddConstraint( a[k] <= 1 )

        # Add (binary) variables b_i = b_s(t) representing the total
        # flow through each node in the graph
        b = self.mp.NewContinuousVariables(nV, 'b')

        # Add continuous variables x, u corresponding to the state and
        # control input for each node
        X = self.mp.NewContinuousVariables(nV, self.n, 'x')
        U = self.mp.NewContinuousVariables(nV, self.m, 'u')

        # Add the spatial (bilinear) variables
        #
        #    y_start_ij = a_ij*y_i
        #    y_end_ij = a_ij*y_j
        #
        # where y_i = [x_i;u_i]
        Y_start = self.mp.NewContinuousVariables(nE, self.n+self.m, 'y_start')
        Y_end = self.mp.NewContinuousVariables(nE, self.n+self.m, 'y_end')

        # Set the perspective-based cost function (13a)
        for e, (i,j) in enumerate(E):
            # Unpack some variables
            a_ij = a[e]
            y_start_ij = Y_start[e,:]
            y_end_ij = Y_end[e,:]

            # Write as
            #   min  f_tilde(lmbda, x)
            #   s.t. g_tilde(lmbda, x) = 0
            # where
            #  f(x) = x'Hx
            # and
            #  g(x) = Gx - g 
            lmbda = a_ij
            x = np.hstack([y_start_ij, y_end_ij])
            H = sp.linalg.block_diag(self.Q, self.R, 0*np.eye(self.n+self.m))  # TODO: add terminal cost

            G = np.block([self.A, self.B, -np.eye(self.n), np.zeros(self.B.shape)])
            g = np.zeros(self.n)

            add_LCQ_perspective_cost(self.mp, H, G, g, x, lmbda)

        # Add the bilinear envelope constraints (13b-c)
        # TODO: check boundedness first
        # TODO: abstract as separate function?
        for e, (i,j) in enumerate(E):
            # Unpack some variables
            a_ij = a[e]
            y_start_ij = Y_start[e,:]
            y_end_ij = Y_end[e,:]
            y_i = np.hstack([X[i,:], U[i,:]])
            y_j = np.hstack([X[j,:], U[j,:]])

            # (a_ij, y_i, y_start_ij) \in Lambda_i
            add_perspective_constraint(self.mp, P[i], y_start_ij, a_ij)
            add_perspective_constraint(self.mp, P[i], y_i - y_start_ij, 1-a_ij)
            
            # (a_ij, y_j, y_end_ij) \in Lambda_j
            add_perspective_constraint(self.mp, P[j], y_end_ij, a_ij)
            add_perspective_constraint(self.mp, P[j], y_j - y_end_ij, 1-a_ij)

        # Add the flow constraints (13d)
        for i, (t,s) in enumerate(V):
            Oi = [k for k, e in enumerate(E) if e[0] == i]
            Ii = [k for k, e in enumerate(E) if e[1] == i]

            a_O = sum(a[k] for k in Oi)
            a_I = sum(a[k] for k in Ii)

            delta_si = 1 if t == 0 and P[s] == start else 0
            delta_ti = 1 if t == self.T-1 and P[s] == target else 0

            self.mp.AddLinearConstraint( a_O - a_I == delta_si - delta_ti )

            # Define b_i as total flow thru each node
            if t == 0:
                self.mp.AddLinearConstraint( b[i] == a_O )
            else:
                self.mp.AddLinearConstraint( b[i] == a_I )

        # Add additional relaxation-tightening constraints
        for i, (t,s) in enumerate(V):
            
            Oi = [k for k, e in enumerate(E) if e[0] == i]
            Ii = [k for k, e in enumerate(E) if e[1] == i]

            a_O = sum(a[k] for k in Oi)
            a_I = sum(a[k] for k in Ii)

            delta_si = 1 if t == 0 and P[s] == start else 0
            delta_ti = 1 if t == self.T-1 and P[s] == target else 0

            # degree constraints (20)
            #if len(Oi) > 0:
            #    self.mp.AddLinearConstraint( a_O <= 1 - delta_ti )
            #if len(Ii) > 0:
            #    self.mp.AddLinearConstraint( a_I <= 1 - delta_si )

            # spatial conservation-of-flow constraints (19)
            #y_start_O = sum(Y_start[k,:] for k in Oi)
            #y_end_I = sum(Y_end[k,:] for k in Ii)
            #
            #ys = np.hstack([X[v0,:], U[v0,:]])
            #yt = np.hstack([X[vF,:], U[vF,:]])

            #self.mp.AddLinearConstraint(eq( y_start_O - y_end_I, delta_si*ys - delta_ti*yt ))

            # spatial degree constraints
            #if len(Oi) > 0:
            #    y_i = np.hstack([X[i,:],U[i,:]])
            #    argument = y_i - delta_ti*yt - y_start_O
            #    scaling = 1 - delta_ti - a_O

            #    add_perspective_constraint(self.mp, P[i], argument, scaling)

            #if len(Ii) > 0:
            #    y_i = np.hstack([X[i,:],U[i,:]])
            #    argument = y_i - delta_si*ys - y_end_I
            #    scaling = 1 - delta_si - a_I

            #    add_perspective_constraint(self.mp, P[i], argument, scaling)

        # Initial condition constraints
        self.mp.AddLinearConstraint(eq( X[v0,:], x0 ))

        # DEBUG: save stuff for later
        self.a = a
        self.b = b
        self.X = X
        self.U = U
        self.Y_start = Y_start
        self.Y_end = Y_end
        self.P = P
        self.V = V
        self.E = E

    def ConstructPartitions(self):
        """
        Define a set of Polytope partitions P_l such that the same predicates hold
        for all values within each partition. 

        @returns lst    A list of Polytopes representing each partition.
        """
        start_time = time.time()
        # Generate list of predicates that establish bounds on state and control input
        bounding_predicates = self.GetBoundingPredicates(self.spec)

        # Create a partition describing all of the bounds on y
        C = np.full((len(bounding_predicates),self.d), np.nan)
        d = np.full((len(bounding_predicates),), np.nan)
        for i, pred in enumerate(bounding_predicates):
            C[i,:] = -pred.A  # polytopes defined as C*y <= d, but
            d[i] = -pred.b    # predicates defined as A*y >= b
        bounding_polytope = Polytope(self.d, ineq_matrices=(C,d)) 
        bounds = Partition(bounding_polytope, [])

        # Check that the bounding poltyope is non-empty.
        assert not bounds.polytope.is_empty(), "Bounding polytope is empty: infeasible specification"
        
        # Check that the bounding polytope is compact (this is needed for the perspective
        # function-based encodings)
        assert bounds.polytope.is_bounded(), "Unbounded specification. Consider adding constraints like G_[0,T] state_bounded"

        # Generate list of all non-bounding predicates
        predicates = [p for p in self.GetPredicates(self.spec) if not p in bounding_predicates]

        # Create partitions
        partition_list = [bounds]
        for p in predicates:
            partition_list = self.SplitAllPartitions(partition_list, p)

        print("Created %s partitions in %0.4fs" % (len(partition_list), time.time()-start_time))

        return partition_list

    def SplitAllPartitions(self, partition_list, pred):
        """
        Given a list of Partitions and a predicate, generate a list of new
        partitions such that the value of the predicate is the same across 
        each new partition. 

        @param partition_list   A list of Partitions
        @param pred             The STLPredicate to split on 

        @returns new_partition_list A new list of Partitions
        """
        new_partition_list = []
        for partition in partition_list:
            new_partition_list += self.SplitPartition(partition, pred)
        return new_partition_list

    def SplitPartition(self, partition, pred):
        """
        Given a (bounded) partition and a (linear) predicate, generate
        new partitions such that the value of the predicate is the same
        accross new partitions. 

        @param partition    The Partition that we'll split
        @param pred         The STLPredicate that we'll use to do the splitting

        @returns partition_list     A list of new Partitions
        """
        assert isinstance(partition, Partition)
        assert isinstance(pred, STLPredicate)

        # Check if this predicate intersects the given partition. If it 
        # doesn't, we can simply return the original partition.
        pred_redundant = partition.polytope.check_ineq_redundancy(-pred.A, -pred.b)
        negation_redundant = partition.polytope.check_ineq_redundancy(pred.A, pred.b)
        redundant = pred_redundant or negation_redundant
        if redundant: return [partition]

        # Create two new partitions based on spliting with the predicate
        pred_poly = Polytope(self.d, ineq_matrices=(-pred.A, -pred.b))
        not_pred_poly = Polytope(self.d, ineq_matrices=(pred.A, pred.b))

        P1 = Partition(partition.polytope.intersection(pred_poly), partition.predicates + [pred])
        P2 = Partition(partition.polytope.intersection(not_pred_poly), partition.predicates)

        return [P1, P2]

    def ConstructGraph(self):
        """
        Create a graph G = (V, E) where each vertex/node corresponds to a partition
        at a given timestep and edges connect all the partitions going forward in time.

        @returns V  A list of vertices. Each vertex is represented by a tuple 
                    i = (t, s) containing the timestep (t) and the index of the 
                    corresponding partition (s).
        @returns E  A list of edges. Each edge is represented by a tuple (i,j)
                    with the indices of the start and end vertices. 

        @note self.ConstructPartitions must be run first.
        """
        V = []
        for t in range(self.T):
            for s in range(len(self.partition_list)):
                V.append((t,s))

        E = []
        for t in range(self.T-1):
            for s in range(len(self.partition_list)):
                for s_prime in range(len(self.partition_list)):
                    E.append(( V.index((t,s)), V.index((t+1,s_prime)) ))

        return (V, E)

    def GetPredicates(self, spec):
        """
        Return a list of all the predicates involved in the given specification.

        @param spec     An STLFormula representing a specification. 
        @returns lst    A list of Predicates p_i which compose the specification spec.
        """
        lst = []

        if isinstance(spec, STLPredicate):
            lst.append(spec)
            return lst
        else:
            for subformula in spec.subformula_list:
                predicates = self.GetPredicates(subformula)
                for predicate in predicates:
                    if predicate not in lst:
                        lst.append(predicate)
            return lst

    def GetNonBoundingStateFormulas(self, spec, bounding_predicates):
        """
        Return a list of state formulas, not including those that simply
        establish bounds on the signal y.
        """
        lst = []
        if spec.is_state_formula():
            predicates = self.GetPredicates(spec)
            if not all([p in bounding_predicates for p in predicates]):
                lst.append(spec)
            return lst
        else:
            for subformula in spec.subformula_list:
                state_formulas = self.GetNonBoundingStateFormulas(subformula, bounding_predicates)
                for formula in state_formulas:
                    if formula not in lst:
                        lst.append(formula)
            return lst

    def GetBoundingPredicates(self, spec, got_always=False):
        """
        Given a specification, return the constraints that describe the convex set
        in which the the solution y must remain. These constraints have the following
        properties:

            - They are added to the top-level specification via "and" operators
            - The temporal operator is "always" across the whole time horizon
            - The "always" operator acts on a state-formula with conjuction only.

        For example, if the specification is

            G_[0,T] ((0 < y) and (y < 2)) and F_[0,T] (y > 1),

        the bounding constraints are (0 < y < 2).

        @param spec         The specification to consider
        @param got_always   A flag for recursion to tell us whether G_[0,T] has been
                            encountered yet

        @returns pred_lst   A list of STLPredicates that all hold across the whole
                            time horizon. 
        """
        lst = []

        if isinstance(spec, STLPredicate):
            if got_always:
                lst.append(spec)
            return lst
        else:
            # Several conditions need to be met for us to continue recursively
            # parsing a subformula: 
            #
            # - the combination type needs to be "and"
            # - the timesteps must be a single timestep or [0,T]
            # - at some point we must have a conjuction over [0,T] (i.e. "always" is applied)
            if (spec.combination_type == "and"):
                if all(t==spec.timesteps[0] for t in spec.timesteps):
                    for subformula in spec.subformula_list:
                        predicates = self.GetBoundingPredicates(subformula, got_always=got_always)
                        for predicate in predicates:
                            if predicate not in lst:
                                lst.append(predicate)
                    return lst
                elif spec.timesteps == [i for i in range(self.T)]:
                    for subformula in spec.subformula_list:
                        predicates = self.GetBoundingPredicates(subformula, got_always=True)
                        for predicate in predicates:
                            if predicate not in lst:
                                lst.append(predicate)
                    return lst
                else:
                    return []
            else:
                return []

    def Solve(self):
        """
        Solve the optimization problem and return the optimal values of (x,u).
        """
        print("Solving...")
        solver = MosekSolver()
        #solver = GurobiSolver()
        res = solver.Solve(self.mp)

        solve_time = res.get_solver_details().optimizer_time
        print("Solve time: ", solve_time)

        if res.is_success():
            # Extract the solution
            a = res.GetSolution(self.a)
            b = res.GetSolution(self.b)
            X = res.GetSolution(self.X)
            U = res.GetSolution(self.U)
            Y_start = res.GetSolution(self.Y_start)
            Y_end = res.GetSolution(self.Y_end)

            # Preallocate y = [x;u] 
            x = np.full((self.n, self.T), 0.0)
            u = np.full((self.m, self.T), 0.0)
            y = np.full((self.n+self.m, self.T), 0.0)

            for i, (t,s) in enumerate(self.V):
                x[:,t] += X[i,:]*b[i]
                u[:,t] += U[i,:]*b[i]
                if t == 0:
                    Oi = [k for k, e in enumerate(self.E) if e[0] == i]
                    y[:,t] += sum(Y_start[k] for k in Oi)
                else:
                    Ii = [k for k, e in enumerate(self.E) if e[1] == i]
                    y[:,t] += sum(Y_end[k] for k in Ii)

            #x = y[:self.n,:]
            #u = y[self.n:,:]

            # Sanity check
            for t in range(self.T-1):
                print(y[:self.n,t+1])
                print(x[:,t+1])
                print(self.A@x[:,t] + self.B@u[:,t])
                print("")

            return x, u
        else:
            print(res.get_solver_details().rescode)
            print(res.get_solver_details().solution_status)
            return None, None

    def plot_partitions(self, show=True):
        """
        Make plot of the projection of all partitions to 2d. 
        """
        for partition in self.partition_list:
            partition.plot(edgecolor='k')

        if show: 
            plt.show()
