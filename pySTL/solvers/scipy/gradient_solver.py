import numpy as np
import time
from scipy.optimize import minimize
from ..base import STLSolver

class ScipyGradientSolver(STLSolver):
    """
    Given an :class:`.STLFormula` :math:`\\varphi` and a :class:`.NonlinearSystem`,
    solve the optimization problem

    .. math::

        \min &  - \\rho^{\\varphi}(y_0,y_1,\dots,y_T) + \sum_{t=0}^T x_t^TQx_t + u_t^TRu_t

        \\text{s.t. } & x_0 \\text{ fixed}

        & x_{t+1} = f(x_t, u_t)

        & y_{t} = g(x_t, u_t)

    using a shooting method and the
    `scipy.optimize <https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html>`_ solver.

    .. warning::

        This solver uses finite-differences to approximate the gradient of the (non-smooth) cost.
        As such, this method is likely to scale extremely poorly.

    :param spec:    An :class:`.STLFormula` describing the specification.
    :param sys:     A :class:`.NonlinearSystem` describing the system dynamics.
    :param x0:      A ``(n,1)`` numpy matrix describing the initial state.
    :param T:       A positive integer fixing the total number of timesteps :math:`T`.
    :param method:  (optional) String characterizing the optimization algorithm to use. See
                    `the scipy docs <https://docs.scipy.org/doc/scipy/reference/reference/generated/scipy.optimize.minimize.html>`_
                    for more details. Default is Sequential Least Squares (``"slsqp"``).
    :param verbose: (optional) A boolean indicating whether to print detailed
                    solver info. Default is ``True``.
    """
    def __init__(self, spec, sys, x0, T, method="slsqp", verbose=True):
        super().__init__(spec, sys, x0, T, verbose)
        self.Q = np.zeros((sys.n,sys.n))
        self.R = np.zeros((sys.m,sys.m))
        self.method = method

    def AddControlBounds(self, u_min, u_max):
        raise NotImplementedError("This solver does not support control bounds!")

    def AddStateBounds(self, x_min, x_max):
        raise NotImplementedError("This solver does not support state bounds!")

    def AddDynamicsConstraints(self):
        raise NotImplementedError("Dynamics constraints are added automatically in cost function computation")

    def AddQuadraticCost(self, Q, R):
        assert Q.shape == (self.sys.n, self.sys.n), "Q must be an (n,n) numpy array"
        assert R.shape == (self.sys.m, self.sys.m), "R must be an (m,m) numpy array"
        self.Q = Q
        self.R = R

    def AddRobustnessCost(self):
        raise NotImplementedError("Robustness cost is added automatically in cost function computation")

    def AddRobustnessConstraint(self):
        raise NotImplementedError("This solver does not support robustness constraints")

    def AddSTLConstraints(self):
        raise NotImplementedError("STL constraints are added automatically in cost function computation")

    def Solve(self):
        # Set an initial guess
        np.random.seed(0)  # for reproducability
        u_guess = np.random.uniform(-0.2,0.2,(self.sys.m,self.T))

        # Run scipy's minimize
        start_time = time.time()
        res = minimize(self.cost, u_guess.flatten(),
                method=self.method)
        solve_time = time.time() - start_time

        if self.verbose:
            print(res.message)
            print("Solve Time: ", solve_time)

        if res.success:
            u = res.x.reshape((self.sys.m,self.T))
            x, y = self.forward_rollout(u)

            rho = self.spec.robustness(y, 0)[0]
            if self.verbose:
                print("Optimal robustness: ", rho)
        else:
            x = None
            u = None
            rho = -np.inf

        return (x,u,rho,solve_time)

    def forward_rollout(self, u):
        """
        Given a control trajectory u of size (m,T),
        perform a forward rollout to compute the associated
        state and output trajectories.
        """
        T = u.shape[1]
        x = np.full((self.sys.n,T),np.nan)
        y = np.full((self.sys.p,T),np.nan)

        x[:,0] = self.x0

        for t in range(T-1):
            x[:,t+1] = self.sys.f(x[:,t], u[:,t])
            y[:,t] = self.sys.g(x[:,t], u[:,t])

        y[:,T-1] = self.sys.g(x[:,t-1], u[:,t-1])

        return x, y

    def cost(self, u_flat):
        """
        Compute the cost (negative robustness) associated
        with the (flattened) control sequence u.
        """
        cost = 0
        # Reconstruct the (m,T) control trajectory from the flattened
        # input. We use a flattened input because scipy's minimize
        # (and indeed most optimization software) assumes the decision
        # variables are put in a vector.
        u = u_flat.reshape((self.sys.m, self.T))

        # Do a forward rollout to compute the state and output trajectories
        x, y = self.forward_rollout(u)

        # Add additional state and control costs
        for t in range(self.T):
            cost += x[:,t].T@self.Q@x[:,t] + u[:,t].T@self.R@u[:,t]

        # Add the (negative) robustness of this signal y with respect
        # to the specification to the cost
        cost += -self.spec.robustness(y,0)

        return cost
