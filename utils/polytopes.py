import numpy as np

class Polytope():
    """
    Basic tools for representing a polytope using the halfspace form

        P = { x | Ax = b
                | Cx <= d }

    and performing common operations. 
    """
    def __init__(self, n, eq_matrices=None, ineq_matrices=None):
        """
        Construct the polytope
        
            P = { x | Ax = b
                    | Cx <= d }

        where x is a vector in R^n.

        @param n                The size of the vector x (a positive integer)
        @param eq_matrices      (optional) A tuple containing (A, b)
        @param ineq_matrices    (optional) A tuple containing (C, d)
        """
        assert isinstance(n, int) and n > 0, "n must be a positive integer"
        self.n = n

        if eq_matrices is not None:
            self.A, self.b = eq_matrices
            m = self.A.shape[0]
            assert self.A.shape == (m,self.n) , "A must have shape (m,n)"
            assert self.b.shape == (m,), "b must have shape (m,)"
        else:
            self.A = np.zeros((0,self.n))
            self.b = np.zeros(0)
        
        if ineq_matrices is not None:
            self.C, self.d = ineq_matrices
            m = self.C.shape[0]
            assert self.C.shape == (m,self.n) , "C must have shape (m,n)"
            assert self.d.shape == (m,), "d must have shape (m,)"
        else:
            self.C = np.zeros((0,self.n))
            self.d = np.zeros(0)

    def evaluate(self, x):
        pass

    def contains(self, x):
        pass

    def intersection(self, other):
        pass

    def simplify(self):
        pass

    def is_empty(self):
        pass

    def add_perspective_constraint(self, prog, phi, x):
        # Should probably double check compactness first
        pass

    def plot(self, **kwargs):
        pass
