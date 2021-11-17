from systems.nonlinear import NonlinearSystem

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
        pass
