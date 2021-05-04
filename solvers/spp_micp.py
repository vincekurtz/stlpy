from solvers.solver_base import STLSolver
from STL import STLPredicate
from utils import Polytope

import numpy as np
import itertools
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
        partition_list = self.ConstructPartitions()

        # DEBUG
        bounding_predicates = self.GetBoundingStateFormulas(self.spec)
        print("")
        for predicate in bounding_predicates:
            print(predicate)
        
    def ConstructPartitions(self):
        """
        Define a set of Polytope partitions P_l such that the same predicates hold
        for all values within each partition. 

        @returns lst    A list of Polytopes representing each partition.
        """
        pred_lst = self.GetPredicates(self.spec)
        for pred in pred_lst:
            # Make polyopes for this predicate and its negation
            P = Polytope(pred.d, ineq_matrices=(-pred.A,-pred.b))
            not_P = Polytope(pred.d, ineq_matrices=(pred.A,pred.b))

        #print(len(pred_lst))
        #print(2**20)
        #lst = [list(i) for i in itertools.product([0,1],repeat=20)]
        #print(len(lst))

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

    def GetBoundingStateFormulas(self, spec, got_always=False):
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
                        predicates = self.GetBoundingStateFormulas(subformula, got_always=got_always)
                        for predicate in predicates:
                            if predicate not in lst:
                                lst.append(predicate)
                    return lst
                elif spec.timesteps == [i for i in range(self.T)]:
                    for subformula in spec.subformula_list:
                        predicates = self.GetBoundingStateFormulas(subformula, got_always=True)
                        for predicate in predicates:
                            if predicate not in lst:
                                lst.append(predicate)
                    return lst
                else:
                    return []
            else:
                return []
