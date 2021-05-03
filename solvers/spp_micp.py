from solvers.solver_base import STLSolver
from STL import STLPredicate
import numpy as np
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

    def __init__(self, spec, A, B, Q, R, x0, T, X, U):
        """
        Initialize the solver.

        @param spec     An STLFormula describing the specification
        @param A        An (n,n) numpy matrix describing state dynamics
        @param B        A (n,m) numpy matrix describing control input dynamics
        @param Q        A (n,n) matrix specifing a running state cost
        @param R        A (m,m) matrix specifing a running control cost
        @param x0       The initial state of the system.
        @param T        An integer specifiying the number of timesteps.
        @param X        A tuple (xmin, xmax) establishing bounds on x_t
        @param U        A tuple (umin, umax) establishing bounds on u_t
        """
        assert len(X) == 2, "X must be a tuple (xmin, xmax)"
        assert len(U) == 2, "X must be a tuple (xmin, xmax)"
        self.xmin = X[0]
        self.xmax = X[1]
        self.umin = U[0]
        self.umax = U[1]
        assert self.xmin < self.xmax, "xmin must be less than xmax"
        assert self.umin < self.umax, "umin must be less than umax"
        super().__init__(spec, A, B, Q, R, x0, T)

        # Create the drake MathematicalProgram instance that will allow
        # us to interface with a MIP solver like Gurobi or Mosek
        self.mp = MathematicalProgram()

        pred_lst = self.GetPredicates(self.spec)

        for pred in pred_lst:
            print(pred)

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

