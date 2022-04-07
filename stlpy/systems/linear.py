from .nonlinear import NonlinearSystem
import numpy as np

class LinearSystem(NonlinearSystem):
    """
    A linear discrete-time system of the form

    .. math::

        x_{t+1} = A x_t + B u_t

        y_t = C x_t + D u_t

    where

        - :math:`x_t \in \mathbb{R}^n` is a system state,
        - :math:`u_t \in \mathbb{R}^m` is a control input,
        - :math:`y_t \in \mathbb{R}^p` is a system output.

    :param A: A ``(n,n)`` numpy array representing the state transition matrix
    :param B: A ``(n,m)`` numpy array representing the control input matrix
    :param C: A ``(p,n)`` numpy array representing the state output matrix
    :param D: A ``(p,m)`` numpy array representing the control output matrix
    """
    def __init__(self, A, B, C, D):
        self.n = A.shape[1]
        self.m = B.shape[1]
        self.p = C.shape[0]

        # Sanity checks on matrix sizes
        assert A.shape == (self.n, self.n), "A must be an (n,n) matrix"
        assert B.shape == (self.n, self.m), "B must be an (n,m) matrix"
        assert C.shape == (self.p, self.n), "C must be an (p,n) matrix"
        assert D.shape == (self.p, self.m), "D must be an (p,m) matrix"

        # Store dynamics parameters
        self.A = A
        self.B = B
        self.C = C
        self.D = D

        # Dynamics functions
        self.dynamics_fcn = lambda x, u: A@x + B@u
        self.output_fcn = lambda x, u: C@x + D@u

class DoubleIntegrator(LinearSystem):
    """
    A linear system describing a double integrator in :math:`d` dimensions
    with full state and control output:

    .. math::

        A = \\begin{bmatrix} I_{d \\times d}  & I_{d \\times d} \\\ 0_{d \\times d} & I_{d \\times d}  \\end{bmatrix}
        \quad
        B = \\begin{bmatrix} 0_{d \\times d} \\\ I_{d \\times d}  \\end{bmatrix}

    .. math::
        C = \\begin{bmatrix} I_{2d \\times 2d} \\\ 0_{d \\times 2d} \\end{bmatrix}
        \quad
        D = \\begin{bmatrix} 0_{2d \\times d} \\\ I_{d \\times d} \\end{bmatrix}

    :param d: Integer describing the dimensionality of the system
    """
    def __init__(self, d):
        I = np.eye(d)
        z = np.zeros((d,d))

        A = np.block([[I,I],
                      [z,I]])
        B = np.block([[z],
                      [I]])
        C = np.block([[I,z],
                      [z,I],
                      [z,z]])
        D = np.block([[z],
                      [z],
                      [I]])

        LinearSystem.__init__(self, A, B, C, D)
