import numpy as np
from STL.formula import STLFormula

class STLPredicate(STLFormula):
    """
    A (linear) STL predicate :math:`\pi` defined by

    .. math::

        a^Ty_t - b \geq 0

    where :math:`y_t \in \mathbb{R}^d` is the value of the signal 
    at a given timestep :math:`t`, :math:`a \in \mathbb{R}^d`, 
    and :math:`b \in \mathbb{R}`.

    :param a: a numpy array or list representing the vector :math:`a`
    :param b: a list, numpy array, or scalar representing :math:`b`
    """
    def __init__(self, a, b, name=None):
        # Convert provided constraints to numpy arrays
        self.a = np.asarray(a).reshape((-1,1))
        self.b = np.atleast_1d(b)

        # Some dimension-related sanity checks
        assert (self.a.shape[1] == 1), "a must be of shape (d,1)"
        assert (self.b.shape == (1,)), "b must be of shape (1,)"
        
        # Store the dimensionality of y_t
        self.d = self.a.shape[0]

        # A unique string describing this predicate
        self.name = name

    def negation(self):
        if self.name is None:
            newname = None
        else:
            newname = "not " + self.name
        return STLPredicate(-self.a, -self.b, name=newname)

    def robustness(self, y, t):
        assert isinstance(y, np.ndarray), "y must be a numpy array"
        assert isinstance(t, int), "timestep t must be an integer"
        assert y.shape[0] == self.d, "y must be of shape (d,T)"
        assert y.shape[1] > t, "requested timestep %s, but y only has %s timesteps" % (t, y.shape[1])

        return self.a.T@y[:,t] - self.b

    def is_state_formula(self):
        return True

    def is_disjunctive_state_formula(self):
        return True

    def is_conjunctive_state_formula(self):
        return True

    def __str__(self):
        if self.name is None:
            return "{ Predicate %s*y >= %s }" % (self.a, self.b)
        else:
            return "{ Predicate " + self.name + " }"

# TODO: create a NonlinearPredicate class based on lambda functions
