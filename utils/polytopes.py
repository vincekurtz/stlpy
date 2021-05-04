import numpy as np
from scipy.linalg import null_space
from pydrake.all import MathematicalProgram, GurobiSolver, ge, le, eq

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
        self.eq_matrices = eq_matrices
        self.ineq_matrices = ineq_matrices

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
        """
        Solve a simple LP to determine whether this polytope is empty.

            min  0
            s.t. Ax = b
                 Cx <= d

        @returns empty  a boolean indicating whether or not this poltyope is empty
        """
        print("Checking polytope emptiness")
        prog = MathematicalProgram()
        x = prog.NewContinuousVariables(self.n, 'x')
        if self.eq_matrices is not None:
            prog.AddConstraint( eq(self.A@x, self.b) )
        if self.ineq_matrices is not None:
            prog.AddConstraint( le(self.C@x, self.d) )
        solver = GurobiSolver()
        res = solver.Solve(prog)

        # If there is a feasible solution then the polytope is non-empty
        return not res.is_success()

    def is_bounded(self):
        """
        Determine whether the given polytope is bounded. 
        
        Uses Stiemke's theorem of alternatives to solve a single LP
        which indicates whether the given polytope is bounded. 
        
        See https://math.stackexchange.com/questions/3592971/algorithm-for-checking-if-a-polyhedron-is-bounded

        @returns bounded    A boolean indicating whether or not this polytope is bounded
        
        @note   So far we just consider polytopes with inequality constraints only.
        """
        print("Checking polytope boundedness")
        if self.eq_matrices is not None:
            raise NotImplementedError
        if self.ineq_matrices is None:
            return False

        # Check the kernel. If ker(C) is nonempty, the polytope is unbounded
        ker = null_space(self.C)
        if ker.size != 0:
            return False

        # Solve the LP 
        #
        # min  ||y||_1 
        # s.t. C'y = 0
        #      y >= 1
        prog = MathematicalProgram()
        y = prog.NewContinuousVariables(self.C.shape[0], 'y')
        prog.AddConstraint( ge(y, 1) )
        prog.AddConstraint( eq(self.C.T@y, 0) )
        solver = GurobiSolver()
        res = solver.Solve(prog)

        # If there is a feasible solution then the polytope is bounded
        return res.is_success()






        

    def add_perspective_constraint(self, prog, phi, x):
        # Should probably double check compactness first
        pass

    def plot(self, **kwargs):
        pass
