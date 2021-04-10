import numpy as np
from abc import ABC, abstractmethod

class STLSolver:
    """
    A simple base class defining a common solver interface
    for different STL synthesis methods. 

    We assume that the specification is defined over a system
    trajectory (x,u) of fixed length T, and the system dynamics
    are given by

        x_{t+1} = A*x_t + B*u_t.

    We also assume that an (auxillary) running cost of the form 

        x'Qx + u'Ru

    is provide. 
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
        self.n = A.shape[1]
        self.m = B.shape[1]
        self.d = self.n + self.m
        self.x0 = np.atleast_1d(x0)

        # Some sanity checks
        assert A.shape == (self.n, self.n), "A must be a square matrix"
        assert B.shape == (self.n, self.m), "B must have dimesions (n,m)"
        assert Q.shape == (self.n, self.n), "Q must have dimensions (n,n)"
        assert R.shape == (self.m, self.m), "R must have dimensions (m,m)"
        assert self.d == spec.d, "Specification size must match size of [x;u]"
        assert self.x0.shape == (self.n,), "x0 must be a 1d numpy array of size n"
        assert isinstance(T,int) and T > 0, "T must be a positive integer"

        # Store the relevant data
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.spec = spec
        self.x0 = x0
        self.T = T+1  # needed to be consistent with how we've defined STLFormula

    @abstractmethod
    def Solve(self):
        """
        Solve the STL syntheis problem and return a satisfying
        state (x) and input (u) sequence. 
        """
        pass
