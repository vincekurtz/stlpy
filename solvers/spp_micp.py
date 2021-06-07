from solvers.solver_base import STLSolver
from STL import STLPredicate, STLFormula
from utils import *

import numpy as np
import scipy as sp
import time
from pydrake.all import (MathematicalProgram, Evaluate,
                         GurobiSolver, MosekSolver, 
                         eq, le)
        
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from itertools import cycle

class SPPMICPSolver(STLSolver):
    """
    Given an STLFormula (spec), a system of the form 

        x_{t+1} = A*x_t + B*u_t,
        y_t = [x_t;u_t],

    use a Mixed-Integer Convex Programming encoding based on 
    Shortest Path Planning (SPP) thru graphs of convex sets
    to find a satisfying trajectory
    """
    def __init__(self, spec, A, B, Q, R, x0, T, relaxed=False):
        """
        Initialize the solver.

        @param spec     An STLFormula describing the specification
        @param A        An (n,n) numpy matrix describing state dynamics
        @param B        A (n,m) numpy matrix describing control input dynamics
        @param Q        A (n,n) matrix specifing a running state cost
        @param R        A (m,m) matrix specifing a running control cost
        @param x0       The initial state of the system.
        @param T        An integer specifiying the number of timesteps.
        @param relaxed  (optional) A boolean indicating whether to solve
                        a convex relaxation of the problem. Defaults to False.
        """
        super().__init__(spec, A, B, Q, R, x0, T)

        # Extract global bounds on the signal from the specification
        self.bounding_predicates = self.GetBoundingPredicates(spec)

        # Construct polytopic partitions based on state formulas
        c_formulas, d_formulas = self.GetSeparatedStateFormulas(spec)
        self.state_formulas = c_formulas + d_formulas
        self.partition_list = self.ConstructStateFormulaPartitions()
        self.S = len(self.partition_list)

        # Identify the starting partition
        y0 = np.hstack([x0,np.zeros(self.m)])
        self.s0 = None
        for s in range(self.S):
            if self.partition_list[s].polytope.contains(y0):
                self.s0 = s

        # Flag for whether to use convex relaxation
        self.convex_relaxation = relaxed

        # Construct a graph where each vertiex corresponds to a partition at
        # a given timestep, and edges exist between each pair of subsequent
        # paritions.
        G = self.ConstructGraph()
        self.V = G[0]  # vertices
        self.E = G[1]  # edges
        self.nV = len(self.V)
        self.nE = len(self.E)

        # Create optimization variables
        self.a = self.NewBinaryVariables(self.nE, 'a')  # a_ij = 1 iff edge (ij) is traversed
        self.l = self.mp.NewContinuousVariables(self.nE, 'l')  # Cost associated with each edge
        self.y = self.mp.NewContinuousVariables(self.nV, self.n+self.m)  # continuous state y_i=[x;u]
                                                                         # for each node
        self.y_start = self.mp.NewContinuousVariables(self.nE, self.n+self.m)  # y_start = a_ij*y_i
        self.y_end = self.mp.NewContinuousVariables(self.nE, self.n+self.m)    # y_end = a_ij*y_j

        # Add cost and constraints to the problem
        self.AddBinaryFlowConstraints()
        self.AddBilinearEnvelopeConstraints()
        self.AddDynamicsConstraints()
        #self.AddRunningCost()

        # Add additional constraints which tighten the convex relaxation
        self.AddOccupancyConstraints()
        self.AddDegreeConstraints()
        self.AddSpatialFlowConstraints()

        # Add STL Constraints
        self.AddSTLConstraints()

    def AddSpatialFlowConstraints(self):
        """ 
        Add the (redundant) spatial flow constraints

            y_O - y_I = 0  if 0 < t < T,
                        y  if t = 0 and s = s0,

        where 
            y_O = sum_{outcoing edges} y_start, 
            y_I = sum_{incoming edges} y_end.
        """
        for i, (t,s) in enumerate(self.V):
            Oi = [k for k, e in enumerate(self.E) if e[0] == i]
            Ii = [k for k, e in enumerate(self.E) if e[1] == i]
            y_O = sum(self.y_start[k] for k in Oi)
            y_I = sum(self.y_end[k] for k in Ii)

            if 0 < t and t < self.T-1:
                self.mp.AddLinearConstraint(eq( y_O - y_I, 0 ))
            elif t == 0:
                if s == self.s0:
                    y = self.y[self.V.index((t,s))]
                    self.mp.AddLinearConstraint(eq( y_O - y_I, y ))
                else:
                    self.mp.AddLinearConstraint(eq( y_O - y_I, 0 ))

    def AddSpatialDegreeConstraints(self):
        pass
    
    def AddDegreeConstraints(self):
        """
        Add the (redundant) degree constraints

            a_O <= 1
            a_I <= 1

            sum_s TotalFlow[s,t] = 1            (occupancy constraint)

        to the optimization problem, where a_O and a_I are total outgoing
        and incoming flows for each node. 
 
        """
        for i, (t,s) in enumerate(self.V):
            Oi = [k for k, e in enumerate(self.E) if e[0] == i]
            Ii = [k for k, e in enumerate(self.E) if e[1] == i]
            a_O = sum(self.a[k] for k in Oi)
            a_I = sum(self.a[k] for k in Ii)

            if t < self.T-1:
                if t == 0 and s == self.s0:
                    self.mp.AddLinearConstraint( a_O == 1 )
                else:
                    self.mp.AddLinearConstraint( a_O <= 1 )
            if 0 < t:
                self.mp.AddLinearConstraint( a_I <= 1 )


    def AddOccupancyConstraints(self):
        """
        Add (redundant) constraints enforcing the sum of the total flows
        at each timestep to be equal to one. 
        """
        for t in range(self.T):
            summ = 0
            for s in range(self.S):
                i = self.V.index((t,s))
                b = self.GetTotalFlow(i)
                summ += b
            self.mp.AddLinearConstraint( summ == 1 )

    def AddBinaryFlowConstraints(self):
        """
        Add the constraints

            a_O - a_I = 0  if 0 < t < T,
                        1  if t = 0 and s = s0

        to the optimization problem, where a_O and a_I are total outgoing
        and incoming flows for each node. 
        """
        for i, (t,s) in enumerate(self.V):
            Oi = [k for k, e in enumerate(self.E) if e[0] == i]
            Ii = [k for k, e in enumerate(self.E) if e[1] == i]
            a_O = sum(self.a[k] for k in Oi)
            a_I = sum(self.a[k] for k in Ii)

            if 0 < t and t < self.T-1:
                self.mp.AddLinearConstraint( a_O - a_I == 0 )
            elif t == 0:
                if s == self.s0:
                    self.mp.AddLinearConstraint( a_O - a_I == 1 )
                else:
                    self.mp.AddLinearConstraint( a_O - a_I == 0 )

    def AddBilinearEnvelopeConstraints(self):
        """
        Add the constraints

            (a_ij, y_i, y_start_ij) \in Lambda_i
            (a_ij, y_j, y_end_ij) \in Lambda_j,

        to the optimization problem, where

            (a, y, z) \in Lambda_i

        ensures that z = a*y and y \in P_i if a is binary. 
        """
        for e, (i,j) in enumerate(self.E):
            # (a_ij, y_i, y_start_ij) \in Lambda_i
            self.AddLambdaConstraint(self.a[e], self.y[i], self.y_start[e], i)

            # (a_ij, y_j, y_end_ij) \in Lambda_j,
            self.AddLambdaConstraint(self.a[e], self.y[j], self.y_end[e], j)

    def AddLambdaConstraint(self, a, y, z, i):
        """
        Add the constraint

            (a, y, z) \in Lambda_i, 

        to the optimization problem, where
            
            Lambda_i = { a, y, z | a \in [0,1], 
                                   z \in a*P_i,
                                   y-z \in (1-a)*P_i }

        and P_i is the i^th polytopic partition.

        @param a    A (binary or relaxed) indicator variable
        @param y    A continuous state which we will constrain to P_i
        @param z    The bilinear flow variable z = a*y
        """
        t, s = self.V[i]
        P = self.partition_list[s].polytope

        # Each partition P = {y | C*y <= d}. 
        #
        # So we can implement the contraints (a, y, z) \in Lambda_i as
        #
        #   C*z <= d*a
        #   C*(y-z) <= d*(1-a)
        #
        # (Note that a is already constrained to [0,1])
        self.mp.AddLinearConstraint(le( P.C@z, P.d*a ))
        self.mp.AddLinearConstraint(le( P.C@(y-z), P.d*(1-a) ))

    def AddDynamicsConstraints(self):
        """
        Add the constraints

            (y_start_ij, y_end_ij) \in a_ij*Y_ij,

        to the optimization where

            Y_ij = { y_i, y_j | x_j = A*x_i + B*u_i }

        imposes the (linear) dynamics constraints. 
        """
        # Set the initial condition constraints
        i = self.V.index((0,self.s0))
        x0 = self.y[i][:self.n]

        self.mp.AddLinearConstraint(eq( x0, self.x0 ))

        for e, (i,j) in enumerate(self.E):
            y_start = self.y_start[e]
            y_end = self.y_end[e]

            x_start = y_start[:self.n]
            u_start = y_start[self.n:]
            x_end = y_end[:self.n]

            self.mp.AddLinearConstraint(eq( x_end, self.A@x_start + self.B@u_start ))

    def AddRunningCost(self):
        """
        Adds the running cost

            sum_{ij \in E} l_tilde(a_ij, y_start_ij, y_end_ij)

        to the optimization problem, where l_tilde is the perspective
        of the quadratic cost

            l = x'Qx + u'Ru
        """
        # We'll implement this cost as a linear cost over slack variables
        # along with SOC constraints:
        #
        #   min  l
        #   s.t. a_ij*l >= x_start'*Q*x_start + u_start'*R*u_start
        #                    + x_end'*Q*x_end   (if t=T)
        #        a_ij >= 0
        #        l >= 0
        for e, (i,j) in enumerate(self.E):
            #l = self.mp.NewContinuousVariables(1,'l')[0]
            l = self.l[e]
            a = self.a[e]
            t, s = self.V[j]

            x_start = self.y_start[e][:self.n]
            u_start = self.y_start[e][self.n:]
            x_end = self.y_end[e][:self.n]

            quad_expr = x_start.T@self.Q@x_start + u_start.T@self.R@u_start
            if t == self.T-1:  # add terminal cost
                quad_expr += x_end.T@self.Q@x_end
            
            self.mp.AddCost(l)
            self.mp.AddRotatedLorentzConeConstraint(l, a, quad_expr)

            # DEBUG
            #self.mp.AddCost(a)

    def AddSTLConstraints(self):
        """
        Add the STL constraints

            (x,u) |= specification

        to the optimization problem, via the recursive introduction
        of binary variables for all subformulas in the specification.
        """
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

        If the formula is a state formula, this constraint uses

            z = sum TotalFlow[s,t] over partitions s that satisfy state formula

        which, together with the perspective-based containment constraints,
        ensures that the state formula holds only if z = 1. 

        If the formula is not a state formula, we recursively traverse the
        subformulas associated with this formula, adding new binary 
        variables z_i for each subformula and constraining

            z <= z_i  for all i

        if the subformulas are combined with conjunction (i.e. all 
        subformulas must hold), or otherwise constraining

            z <= sum(z_i)

        if the subformulas are combined with disjuction (at least one
        subformula must hold). 
        """
        if formula in self.bounding_predicates:
            # We can safely ignore any predicates that have to do only
            # with establishing bounds, since the partitioning already
            # accounts for this
            pass

        elif formula in self.state_formulas:
            # For a state formula, we need to add the corresponding constraints
            P_lst, s_lst = self.PartitionsSatisfying(formula)
            total_flow = 0
            for s in s_lst:
                i = self.V.index((t,s))
                total_flow += self.GetTotalFlow(i)
            self.mp.AddConstraint(z[0] == total_flow)

        # We haven't reached the bottom of the tree, so keep adding
        # boolean constraints recursively
        else:
            if formula.combination_type == "and":
                for i, subformula in enumerate(formula.subformula_list):
                    z_sub = self.mp.NewContinuousVariables(1)
                    self.mp.AddConstraint(z_sub[0] <= 1)
                    self.mp.AddConstraint(0 <= z_sub[0])

                    t_sub = formula.timesteps[i]   # the timestep at which this formula 
                                                   # should hold
                    self.AddSubformulaConstraints(subformula, z_sub, t+t_sub)

                    # z <= z_i
                    self.mp.AddConstraint( z[0] <= z_sub[0] )

            else:  # combination_type == "or":
                z_subs = []
                for i, subformula in enumerate(formula.subformula_list):
                    z_sub = self.mp.NewContinuousVariables(1)
                    self.mp.AddConstraint(z_sub[0] <= 1)
                    self.mp.AddConstraint(0 <= z_sub[0])

                    t_sub = formula.timesteps[i]
                    z_subs.append(z_sub)
                    self.AddSubformulaConstraints(subformula, z_sub, t+t_sub)

                # z <= sum(z_subs)
                A = np.hstack([1,-np.ones(len(z_subs))])[np.newaxis]
                lb = -np.array([np.inf])
                ub = np.array([0])
                vars = np.vstack([z,z_subs])
                self.mp.AddLinearConstraint(A=A, lb=lb, ub=ub, vars=vars)

    def PartitionsSatisfying(self, state_formula):
        """
        Return a list of all of the partitions that satisfy the given state formula. 

        @param state_formula    The STLFormula under consideration

        @returns Ps   A list of polytope Partitions P that satisfy the predicate
        @returns ss   A the indices of these same partitions

        @note self.ConstructPartitions must be called first
        """
        Ps = [P for s, P in enumerate(self.partition_list) if state_formula in P.formulas]
        ss = [s for s, P in enumerate(self.partition_list) if state_formula in P.formulas]
        return Ps, ss

    def ConstructStateFormulaPartitions(self):
        """
        Define a set of Polytope partitions such that the same state formulas 
        hold accross each partition.

        @returns lst    A list of Partitions representing each partition.

        @note self.state_formulas must be defined first
        @note self.bounding_predicates must be defined first
        """
        start_time = time.time()

        # Create a partition describing all of the bounds on y
        bounding_polytope = self.GetBoundingPolytope(self.bounding_predicates)
        bounds = Partition(bounding_polytope, [])

        # Generate list of all non-bounding predicates
        predicates = [p for p in self.GetPredicates(self.spec) if not p in self.bounding_predicates]

        # Get labeled polytopes corresponding to all the state formulas
        # (These are Partition objects, but will not necessarily be included in the
        # partition list, since they might overlap)
        state_partitions = []
        for formula in self.state_formulas:
            poly = self.ConstructStateFormulaPolytope(formula, bounding_polytope)
            state_partitions.append(Partition(poly, [formula]))

        # Create partitions
        partition_list = [bounds]
        for p in predicates:
            partition_list = self.StateSplitAllPartitions(partition_list, p, state_partitions)

        print("Created %s partitions in %0.4fs" % (len(partition_list), time.time()-start_time))

        return partition_list

    def HoldsOverPolytope(self, poly, state_partition):
        """
        Check whether a given state formula holds everywhere, nowhere, 
        or only in some places over the given polytope. 

        @param poly             The polytope that we want to test
        @param state_partition  A Partition object containing labeled with the
                                state formula in question, and including corresponding
                                polytopic region.

        @returns status     A string indicating the result. Can be "everywhere",
                            "nowhere" or "some_places"
        """
        intersection = poly.intersection(state_partition.polytope)
        intersection.simplify()

        if (not intersection.is_bounded()) or intersection.is_empty():
            if state_partition.formulas[0].combination_type == "and":
                return "nowhere"
            return "everywhere"

        if intersection == poly:
            if state_partition.formulas[0].combination_type == "and":
                return "everywhere"
            return "nowhere"

        return "some_places"

    def ConstructPartitions(self):
        """
        Define a set of Polytope partitions P_l such that the same predicates hold
        for all values within each partition. 

        @returns lst    A list of Partitions representing each partition.
        """
        start_time = time.time()

        # Create a partition describing all of the bounds on y
        bounding_predicates = self.GetBoundingPredicates(self.spec)
        bounding_polytope = self.GetBoundingPolytope(bounding_predicates)
        bounds = Partition(bounding_polytope, bounding_predicates)

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

        P1 = Partition(partition.polytope.intersection(pred_poly), partition.formulas + [pred])
        P2 = Partition(partition.polytope.intersection(not_pred_poly), partition.formulas)

        return [P1, P2]

    def StateSplitAllPartitions(self, partition_list, pred, state_partitions):
        """
        Given a list of Partitions and a predicate, generate a list of new
        partitions such that the value of all the state formulas is the same across 
        each new partition. 

        @param partition_list   A list of Partitions
        @param pred             The STLPredicate to split on 
        @param state_partitions A list of Partition objects containing each state
                                formula and the corresponding polytope

        @returns new_partition_list A new list of Partitions
        """
        new_partition_list = []
        for partition in partition_list:
            new_partition_list += self.StateSplitPartition(partition, pred, state_partitions)
        return new_partition_list

    def StateSplitPartition(self, partition, pred, state_partitions):
        """
        Given a (bounded) partition and a (linear) predicate, generate
        new partitions such that the values of all state formulas are the same
        accross new partitions. 

        @param partition        The Partition that we'll split on
        @param pred             The STLPredicate that we'll use to do the splitting
        @param state_partitions A list of Partition objects containing each state
                                formula and the corresponding polytope

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
        poly_one = Polytope(self.d, ineq_matrices=(-pred.A, -pred.b)).intersection(partition.polytope)
        poly_two = Polytope(self.d, ineq_matrices=(pred.A, pred.b)).intersection(partition.polytope)

        part_one = Partition(poly_one, [])
        part_two = Partition(poly_two, [])

        # Determine the state formulas that hold over each polytope
        part_one_everywhere = []
        part_one_nowhere = []
        part_two_everywhere = []
        part_two_nowhere = []

        for sp in state_partitions:
            state_formula = sp.formulas[0]
            
            p1_status = self.HoldsOverPolytope(poly_one, sp)
            if p1_status == "everywhere":
                part_one_everywhere.append(state_formula)
            elif p1_status == "nowhere":
                part_one_nowhere.append(state_formula)

            p2_status = self.HoldsOverPolytope(poly_two, sp)
            if p2_status == "everywhere":
                part_two_everywhere.append(state_formula)
            elif p2_status == "nowhere":
                part_two_nowhere.append(state_formula)

        part_one.formulas = part_one_everywhere
        part_two.formulas = part_two_everywhere

        # Check whether labels of both partitions are identical. If they are, we
        # don't need to split. 
        labels_match = (part_one_everywhere == part_two_everywhere) and \
                (part_one_nowhere == part_two_nowhere)
        if labels_match:
            return [partition]
        
        # Otherwise, we split and return the two partitions
        return [part_one, part_two]

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
                    if isinstance(formula, STLFormula) and len(formula.subformula_list) == 1:
                        # This happens sometimes with the way we've encoded 'until',
                        # results in duplicate state formulas for partitioning
                        formula = formula.subformula_list[0]
                    if formula not in lst:
                        lst.append(formula)
            return lst
    
    def GetSeparatedStateFormulas(self, spec):
        """
        Return lists of conjunctive and disjuctive state formulas, not including 
        those that simply establish bounds on the signal y.

        @param spec                 The specification to parse

        @returns conjunction_list   A list of conjunctive state formulas
        @returns disjunction_list   A list of disjuctive state formulas

        @note self.bounding_predicates must be set first
        """
        c_list = []   # conjuction 
        d_list = []  # disjuction

        # The given specification is itself conjunctive state formula
        if spec.is_conjunctive_state_formula():
            predicates = self.GetPredicates(spec)
            if not all([p in self.bounding_predicates for p in predicates]):
                c_list.append(spec)
            return c_list, d_list

        # The given specification is a itself a disjunctive state formula
        elif spec.is_disjunctive_state_formula():
            predicates = self.GetPredicates(spec)
            if not all([p in self.bounding_predicates for p in predicates]):
                d_list.append(spec)
            return c_list, d_list

        # The given specification is neither a conjunctive nor a disjuctive state
        # formula, so we need to keep parsing recursively to find the state formulas
        else:
            for subformula in spec.subformula_list:
                c_formulas, d_formulas = self.GetSeparatedStateFormulas(subformula)
                for formula in c_formulas:
                    if isinstance(formula, STLFormula) and len(formula.subformula_list) == 1:
                        # This happens sometimes with the way we've encoded 'until',
                        # results in duplicate state formulas for partitioning
                        formula = formula.subformula_list[0]
                    if formula not in c_list:
                        c_list.append(formula)
                for formula in d_formulas:
                    if isinstance(formula, STLFormula) and len(formula.subformula_list) == 1:
                        # This happens sometimes with the way we've encoded 'until',
                        # results in duplicate state formulas for partitioning
                        formula = formula.subformula_list[0]
                    if formula not in d_list:
                        d_list.append(formula)
            return c_list, d_list

    def GetBoundingPredicates(self, spec, got_always=False):
        """
        Given a specification, return the constraints that describe the convex set
        in which the the solution y must remain. These constraints have the following
        properties:

            - They are added to the top-level specification via "and" operators
            - The temporal operator is "always" across the whole time horizon
            - The "always" operator acts on a state-formula with conjunction only.

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
            # - at some point we must have a conjunction over [0,T] (i.e. "always" is applied)
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
       
    def GetBoundingPolytope(self, bounding_predicates):
        """
        Construct a polytope that bounds the signal y for all time

        @param bounding_predicates  A list of STLPredicates that establish bounds on y
                                    (see self.GetBoundingPredicates)

        @returns poly   A Polytope that bounds the y for all time.
        """
        C = np.full((len(bounding_predicates),self.d), np.nan)
        d = np.full((len(bounding_predicates),), np.nan)
        for i, pred in enumerate(bounding_predicates):
            C[i,:] = -pred.A  # polytopes defined as C*y <= d, but
            d[i] = -pred.b    # predicates defined as A*y >= b
        poly = Polytope(self.d, ineq_matrices=(C,d)) 
        
        # Check that the bounding poltyope is non-empty.
        assert not poly.is_empty(), "Bounding polytope is empty: infeasible specification"
        
        # Check that the bounding polytope is compact (this is needed for the perspective
        # function-based encodings)
        assert poly.is_bounded(), "Unbounded specification. Consider adding constraints like G_[0,T] state_bounded"

        return poly

    def ConstructStateFormulaPolytope(self, formula, bounding_poly):
        """
        Given a (conjunctive or disjunctive) state formula, construct a
        corresponding polytope. 

        @param formula          The STLFormula in question
        @param bounding_poly    A Polytope establishing global bounds on the signal y

        @returns poly   A Polytope such that the given state formula holds everywhere 
                        (in the case of conjunction) or nowhere (in the case of disjuction)
                        within this polytope. 
        """
        pred = self.GetPredicates(formula)

        C = np.full((len(pred),self.d),np.nan)
        d = np.full((len(pred),),np.nan)

        if formula.combination_type == "and":  # conjunctive type
            for i, p in enumerate(pred):
                C[i,:] = -p.A
                d[i] = -p.b
        elif formula.combination_type == "or":  # disjuctive type
            for i, p in enumerate(pred):
                C[i,:] = p.A
                d[i] = p.b
        else:
            raise ValueError("Unknown combination type %s" % formula.combination_type)

        poly = Polytope(self.d, ineq_matrices=(C,d)).intersection(bounding_poly)

        assert poly.is_bounded(), "Unbounded polytope: make sure all state formulas are purely conjunctive and disjuctive"
       
        return poly

    def GetTotalFlow(self, i):
        """
        Given the index i of a node in the graph, return a Drake expression corresponding
        to the total flow b_i through that node, where

            b_i = sum_{ j in Oi } a_ij  if t = 0
                  sum_{ j in Ii } a_ji  otherwise

        @param  i   The index of the node in question (integer in [0,nV])

        @returns b_i    A Drake expression corresponding to the total flow thru node i.
        """
        assert 0 <= i and i <= self.nV, "Asked for node %i, but must be in [0,%s]" % (i,self.nV)
        t, s = self.V[i]

        if t == 0:
            Oi = [k for k, e in enumerate(self.E) if e[0] == i]
            a_O = sum(self.a[k] for k in Oi)
            return a_O
        else:
            Ii = [k for k, e in enumerate(self.E) if e[1] == i]
            a_I = sum(self.a[k] for k in Ii)
            return a_I

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
            self.mp.SetSolverOption(solver.solver_id(), "OutputFlag",1)

        res = solver.Solve(self.mp)

        solve_time = res.get_solver_details().optimizer_time
        print("Solve time: ", solve_time)

        if res.is_success():
            y = np.full((self.n+self.m, self.T), 0.0)
            b = np.full((self.S, self.T), np.nan)

            for i in range(self.nV):
                t, s = self.V[i]
                
                bi = res.GetSolution(self.GetTotalFlow(i))
                yi = res.GetSolution(self.y[i,:])
                byi = Evaluate(bi*yi).flatten()

                y[:,t] += byi
                b[s,t] = Evaluate([bi])

            x = y[:self.n,:]
            u = y[self.n:,:]

            print(b)

            rho = self.spec.robustness(y,0)
            print("Optimal Cost: ", res.get_optimal_cost())
            print("Optimal Robustness: ", rho[0])

            # Store the solution. This will allow us to make
            # pretty plots, etc later. 
            self.res = res

        else:
            print("No solution found")
            x = None
            u = None

        return x, u
    
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

    def plot_relaxed_solution(self, show=True):
        """
        Make a plot of the first two dimensions of the signal y_s, 
        with opacity determined by y_s. 

        Recall that y_s is constrained to the s^th partition, and
        y[t] = y_s[t] if b_s[t] = 1. 

        @note self.Solve must be called and return a valid solution first. 
        """
        assert hasattr(self, 'res'), "self.Solve() must return a positive result first"

        for s in range(self.S):
            y = self.res.GetSolution(self.ys[s])
            b = self.res.GetSolution(self.b[s])
            for t in range(self.T):
                plt.scatter(*y[:2,t], marker='^', color='k', alpha=b[t])

        if show:
            plt.show()

    def animate_partition_sequence(self):
        """
        Make an animation of the sequence of partitions corresponding to the 
        solution. 

        Recalling that the signal passes through the s^th partition at time t
        if b_s[t] = 1, we use b_s[t] to determine the opacity of each partition.

        @note self.Solve must be called and return a valid solution first. 
        """
        assert hasattr(self, 'res'), "self.Solve() must return a positive result first"

        fig = plt.gcf()
        ax = plt.gca()

        # Add transparent patches for each partition
        patches = []
        for P in self.partition_list:
            patch = P.plot(ax=ax,edgecolor='k',alpha=0.5)
            patches.append(patch)

        # Get solution values for b_s[t]
        b = np.full((self.S, self.T),np.nan)
        for s in range(self.S):
            for t in range(self.T):
                i = self.V.index((t,s))
                b[s][t] = self.res.GetSolution(self.b[i])

        # Function for sending data to the animation
        def data_gen():
            gen_list = ((t,b[:,t]) for t in range(self.T))
            return gen_list

        # Function that's called at each step of the animation
        def run(data):
            t, b = data
            b = np.clip(b,0,1)  # sometimes (due to numerics) values of b will be just
                                # outside [0,1], so clip them here
            ax.set_title("t = %s" % t)
            for s in range(self.S):
                patches[s].set_alpha(b[s])

        ani = FuncAnimation(fig, run, data_gen, interval=500)

        plt.show()

    def plot_partitions(self, show=True):
        """
        Make plot of the projection of all partitions to 2d. 
        """

        colors = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

        for partition in self.partition_list:
            partition.plot(edgecolor='k',facecolor=next(colors))

        if show: 
            plt.show()
