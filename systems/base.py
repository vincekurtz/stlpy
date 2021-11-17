import numpy as np
from abc import ABC, abstractmethod

class System(ABC):
    """
    An abstract class which represents some (possibly nonlinear)
    discrete-time control system

    .. math::
        
        x_{t+1} = f(x_t, u_t)

        y_t = g(x_t, u_t)

    where

        - :math:`x_t \in \mathbb{R}^n` is a system state,
        - :math:`u_t \in \mathbb{R}^m` is a control input,
        - :math:`y_t \in \mathbb{R}^p` is a system output.
    """
    @abstractmethod
    def next_state(self, x, u):
        """
        Given state :math:`x_t` and control :math:`u_t`, compute
        the forward dynamics
        
        .. math::
            
            x_{t+1} = f(x_t, u_t).

        :param x:   The current state :math:`x_t`
        :param u:   The control input :math:`u_t`

        :return:    The subsequent state :math:`x_{t+1}`
        """
        pass

    @abstractmethod
    def output(self, x, u):
        """
        Given state :math:`x_t` and control :math:`u_t`, compute
        the output
        
        .. math::
            
            y_t = g(x_t, u_t).

        :param x:   The current state :math:`x_t`
        :param u:   The control input :math:`u_t`

        :return:    The output :math:`y_t`
        """
        pass
