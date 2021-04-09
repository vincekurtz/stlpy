##
#
# Tools for gradient-based STL synthesis.
#
##

import numpy as np
from scipy.optimize import minimize

class GradientSolver:
    """
    Given an STLFormula, desired trajectory length T, and
    a system of the form 

        x_{t+1} = A*x_t + B*u_t,
        y_t = [x_t;u_t],

    use gradient-based optimization to find a trajectory (x,u) that satisfies
    the specification. 

    Note that this optimizes over the (non-smooth) robustness measure directly,
    rather than using a smooth approximation. 
    """
    def __init__(self, spec, A, B, T, x0):
        """
        Initialize the solver.

        @param spec     An STLFormula describing the specification
        @param A        An (n,n) numpy matrix describing state dynamics
        @param B        A (n,m) numpy matrix describing control input dynamics
        @param T        An integer specifiying the number of timesteps.
        @param x0       The initial state of the system.
        """
        self.n = A.shape[1]
        self.m = B.shape[1]
        self.d = self.n + self.m
        self.x0 = np.atleast_1d(x0)

        # Some sanity checks
        assert A.shape == (self.n, self.n), "A must be a square matrix"
        assert B.shape == (self.n, self.m), "B must be have dimesions (n,m)"
        assert self.d == spec.d, "Specification size must match size of [x;u]"
        assert self.x0.shape == (self.n,), "x0 must be a 1d numpy array of size n"
        assert isinstance(T,int) and T > 0, "T must be a positive integer"

        # Store the relevant data
        self.A = A
        self.B = B
        self.spec = spec
        self.x0 = x0
        self.T = T+1  # needed to be consistent with how we've defined STLFormula

    def Solve(self):
        """
        Solve the optimization problem 

            max  rho([x;u]) 
            s.t. x_{t+1} = A x_t + B u_t
                 x_0 = x0

        using gradient ascent and return the solution. 
        """
        # Set an initial guess
        u_guess = np.zeros((self.m, self.T))

        # Run scipy's minimize
        res = minimize(self.cost, u_guess.flatten(),
                    method="SLSQP") # sequential quadratic programming
        print(res.message)

        if res.success:
            u = res.x.reshape((self.m,self.T))
            x = self.forward_rollout(u)
        else:
            x = None
            u = None

        return (x,u)

    def forward_rollout(self, u):
        """
        Given a control trajectory u of size (m,T), 
        perform a forward rollout to compute the associated
        state trajectory.
        """
        T = u.shape[1]
        x = np.full((self.n,T),np.nan)
        x[:,0] = self.x0
        
        for t in range(T-1):
            x[:,t+1] = self.A@x[:,t] + self.B@u[:,t]

        return x

    def cost(self, u_flat):
        """
        Compute the cost (negative robustness) associated
        with the (flattened) control sequence u.
        """
        # Reconstruct the (m,T) control trajectory from the flattened
        # input. We use a flattened input because scipy's minimize
        # (and indeed most optimization software) assumes the decision
        # variables are put in a vector. 
        u = u_flat.reshape((self.m, self.T))

        # Do a forward rollout to compute the associated state trajectory
        # and construct the signal y = [x;u]
        x = self.forward_rollout(u)
        y = np.vstack([x,u])
        
        # Return the (negative) robustness of this signal y with respect
        # to the specification (since our goal is to maximize robustness). 
        return -self.spec.robustness(y,0)
