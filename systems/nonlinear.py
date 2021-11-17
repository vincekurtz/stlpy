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
                numpy array (:math:`x_{t+1}`)
    :param g:   A function representing :math:`f`, which takes two numpy 
                arrays (:math:`x_t,u_t`) as input and returns another 
                numpy array (:math:`y_{t}`)
    """
    def __init__(self, f, g):
        # TODO: do some checks on function signature
        self.f = f
        self.g = g
    
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
        return self.f(x, u)

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
        return self.g(x,u)
