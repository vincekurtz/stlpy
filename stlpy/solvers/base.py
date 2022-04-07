from abc import ABC, abstractmethod

class STLSolver(ABC):
    """
    A simple abstract base class defining a common solver interface
    for different optimization-based STL synthesis methods.

    This class considers variations on the trajectory synthesis problem

    .. math::

        \min & \sum_{t=0}^T x_t^TQx_t + u_t^TRu_t

        \\text{s.t. } & x_0 \\text{ fixed }

        & x_{t+1} = f(x_t, u_t)

        & y_t = g(x_t, u_t)

        & \\rho^{\\varphi}(y_0,y_1,\dots,y_T) \geq 0

    where :math:`Q \\succeq 0` and :math:`R \\succeq 0` are cost weights, :math:`f` and :math:`g`
    define the system dynamics, and :math:`\\rho` is the robustness measure associated with
    a given STL specification :math:`\\varphi`.

    Possible variations include using the robustness measure :math:`\\rho^\\varphi`
    as a cost, dropping the quadratic running cost, and removing the hard satisfaction
    constriant :math:`\\rho^{\\varphi}\geq 0`.

    :param spec:    An :class:`.STLFormula` describing the specification.
    :param sys:     An :class:`.NonlinearSystem` characterizing the system dynamics.
    :param x0:      A ``(n,1)`` numpy array representing the initial state :math:`x_0`.
    :param T:       A positive integer fixing the total number of timesteps :math:`T`.
    :param verbose: A boolean specifying whether to print detailed solver info.
    """
    def __init__(self, spec, sys, x0, T, verbose):
        # Store the relevant data
        self.sys = sys
        self.spec = spec
        self.x0 = x0
        self.T = T+1  # needed to be consistent with how we've defined STLFormula
        self.verbose = verbose

    @abstractmethod
    def AddDynamicsConstraints(self):
        """
        Add the dynamics constraints

        .. math::

            & x_0 \\text{ fixed }

            & x_{t+1} = f(x_t, u_t)

            & y_t = g(x_t, u_t)

        to the optimization problem.
        """
        pass

    @abstractmethod
    def AddSTLConstraints(self):
        """
        Add constraints to the optimization problem to define
        the robustness measure

        .. math::

            \\rho^{\\varphi}(y_0,y_1,\dots,y_T).
        """
        pass

    @abstractmethod
    def AddRobustnessCost(self):
        """
        Add the robustness measure as a (linear) cost
        to the optimization problem:

        .. math::

            \min -\\rho^{\\varphi}(y_0,y_1,\dots,y_T).

        """
        pass

    @abstractmethod
    def AddRobustnessConstraint(self, rho_min=0.0):
        """
        Add a constraint on the robustness measure to the
        optimization problem:

        .. math::

            \\rho^{\\varphi}(y_0,y_1,\dots,y_T) \geq \\rho_{min}

        :param rho_min:     (optional) Minimum robustness measure :math:`\\rho_{min}`.
                            Defaults to 0, which enforces STL satisfaction.
        """
        pass

    @abstractmethod
    def AddControlBounds(self, u_min, u_max):
        """
        Add upper and lower bounds on the control inputs :math:`u_t`
        to the optimization problem:

        .. math::

            u_{min} \leq u_t \leq u_{max} \quad \\forall t

        :param u_min:   A ``(m,)`` numpy array specifying the minimum control input
        :param u_max:   A ``(m,)`` numpy array specifying the maximum control input
        """
        pass

    @abstractmethod
    def AddStateBounds(self, x_min, x_max):
        """
        Add upper and lower bounds on the state variables :math:`x_t`
        to the optimization problem:

        .. math::

            x_{min} \leq x_t \leq x_{max} \quad \\forall t

        :param x_min:   A ``(n,)`` numpy array specifying :math:`x_{min}`
        :param x_max:   A ``(n,)`` numpy array specifying :math:`x_{max}`
        """
        pass

    @abstractmethod
    def AddQuadraticCost(self, Q, R):
        """
        Add a quadratic running cost to the optimization problem:

        .. math::

            \min \sum_{t=0}^T x_t^TQx_t + u_t^TRu_t

        :param Q:   A ``(n,n)`` numpy array representing the state penalty matrix :math:`Q`
        :param R:   A ``(m,m)`` numpy array representing the control penalty matrix :math:`R`
        """
        pass

    @abstractmethod
    def Solve(self):
        """
        Solve the STL syntheis optimization problem and return an optimal trajectory.

        :return x:          A ``(n,T)`` numpy array containing the optimal state :math:`x_t`
                            for each timestep.
        :return u:          A ``(m,T)`` numpy array containing the optimal control :math:`x_t`
                            for each timestep.
        :return rho:        A scalar indicating the optimal robustness value.
        :return solve_time: The time it took the solver to find a solution, in seconds.

        .. note::

            ``x`` and ``u`` are returned as ``None`` if the optimization problem is
            infeasible or the solver is unable to find a solution.
        """
        pass

