import numpy as np


class NonlinearSystem:
    """
    A class which represents some (possibly nonlinear)
    discrete-time control system

    .. math::

        x_{t+1} = f(x_t, u_t)

        y_t = g(x_t, u_t)

    where

        - :math:`x_t \in \mathbb{R}^n` is a system state,
        - :math:`u_t \in \mathbb{R}^m` is a control input,
        - :math:`y_t \in \mathbb{R}^p` is a system output.

    :param f:   A function representing :math:`f`, which takes two numpy
                arrays (:math:`x_t,u_t`) as input and returns another
                numpy array (:math:`x_{t+1}`).
    :param g:   A function representing :math:`f`, which takes two numpy
                arrays (:math:`x_t,u_t`) as input and returns another
                numpy array (:math:`y_{t}`).
    :param n:   Size of the state vector :math:`x_t`.
    :param m:   Size of the control vector :math:`u_t`.
    :param p:   Size of the output vector :math:`p_t`.

    """
    def __init__(self, f, g, n, m, p):
        # TODO: do some checks on function signature
        self.dynamics_fcn = f
        self.output_fcn = g
        self.n = n
        self.m = m
        self.p = p

    def f(self, x, u):
        """
        Given state :math:`x_t` and control :math:`u_t`, compute
        the forward dynamics

        .. math::

            x_{t+1} = f(x_t, u_t).

        :param x:   The current state :math:`x_t`
        :param u:   The control input :math:`u_t`

        :return:    The subsequent state :math:`x_{t+1}`
        """
        return self.dynamics_fcn(x, u)

    def g(self, x, u):
        """
        Given state :math:`x_t` and control :math:`u_t`, compute
        the output

        .. math::

            y_t = g(x_t, u_t).

        :param x:   The current state :math:`x_t`
        :param u:   The control input :math:`u_t`

        :return:    The output :math:`y_t`
        """
        return self.output_fcn(x,u)

class Unicycle(NonlinearSystem):
    r"""
    A simple nonlinear system representing a 2D mobile robot with
    unicycle dynamics. The robot is controlled by specifing a forward
    velociy :math:`v` and an angular velocity :math:`\omega`. 

    This is example of a non-holonomic system: the robot cannot
    directly control its motion in the horizontal direction.

    The state is given by 

    .. math::

        x = \begin{bmatrix} p_x \\ p_y \\ \theta \end{bmatrix},

    where :math:`p_x` and :math:`p_y` are positions in the plane and
    :math:`\theta` is an orientation. The dynamics are given by

    .. math::

        \dot{x} = \begin{bmatrix} v \cos(\theta) \\ v \sin(\theta) \\ \omega \end{bmatrix}

    and the control input is :math:`u = \begin{bmatrix} v \\ \omega \end{bmatrix}`.
    We use forward Euler integration to transform this into a discrete-time system:

    .. math::

        x_{t+1} = x_t + \dot{x}~dt.

    The system output is simply the state of the robot, :math:`y_t = x_t`.

    :param dt:  Discretization step size (for forward Euler integration)
    """
    def __init__(self, dt):
        self.dt = dt
        
        # State, control, and output sizes
        self.n = 3
        self.m = 2
        self.p = 3

    def f(self, x, u):
        v = u[0]      # linear velocity
        omega = u[1]  # angular velocity
        theta = x[2]  # orientation

        xdot = np.array([ v * np.cos(theta),
                          v * np.sin(theta),
                          omega])

        return x + self.dt * xdot

    def g(self, x, u):
        return x
