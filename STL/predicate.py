import numpy as np
from STL.formula import STLFormulaBase

class STLPredicate(STLFormulaBase):
    """
    A (linear) stl predicate defined by

        A*y_t - b >= 0

    where y_t is the value of the signal 
    at a given timestep t.
    """
    def __init__(self, A, b, name=None):
        # Convert provided constraints to numpy arrays
        self.A = np.asarray(A)
        self.b = np.atleast_1d(b)
        
        # Some dimension-related sanity checks
        assert (self.A.shape[0] == 1), "A must be of shape (1,d)"
        assert (self.b.shape == (1,)), "b must be of shape (1,)"
        
        # Store the dimensionality of y_t
        self.d = self.A.shape[1]

        # A unique string describing this predicate
        self.name = name

    def robustness(self, y, t):
        assert isinstance(y, np.ndarray), "y must be a numpy array"
        assert isinstance(t, int), "timestep t must be an integer"
        assert y.shape[0] == self.d, "y must be of shape (d,T)"
        assert y.shape[1] > t, "requested timestep %s, but y only has %s timesteps" % (t, y.shape[1])

        return self.A@y[:,t] - self.b

    def __str__(self):
        if self.name is None:
            return "{ Predicate %s*y >= %s }" % (self.A, self.b)
        else:
            return "{ " + self.name + " }"

# TODO: create a NonlinearPredicate class based on lambda functions
