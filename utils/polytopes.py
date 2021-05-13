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

    def contains(self, x):
        """
        Determine whether this polytope contains the given point. 

        @param x    The vector that we would like to test

        @returns    A boolean indicating whether x is in this polytope. 
        """
        assert x.shape == (self.n,), "Vector length doesn't match polytope dimension"

        return np.all(self.A@x == self.b) and np.all(self.C@x <= self.d)

    def intersection(self, other):
        """
        Return a new polytope which represents the intersection of this polytope and
        the another. 

        @param other    The other polytope that we'll intersect this one with. 

        @returns new    A new polytope which is the intersection of this one and other. 
        """
        assert self.n == other.n, "Polytopes must have the same underlying dimension."

        newA = np.vstack([self.A, other.A])
        newb = np.hstack([self.b, other.b])

        newC = np.vstack([self.C, other.C])
        newd = np.hstack([self.d, other.d])

        return Polytope(self.n, eq_matrices=(newA,newb), ineq_matrices=(newC,newd))

    def add_eq_constraint(self, A, b):
        """
        Add the equality constraint A*x = b to this polytope. 

        @param A    A matrix of size (m,n)
        @param b    A vector of size (m,)
        """
        m = A.shape[0]
        n = A.shape[1]
        assert n == self.n, "A must be of size (m,n)"
        assert b.shape == (m,), "b must be of size (m,)"

        self.A = np.vstack([self.A,A])
        self.b = np.hstack([self.b,b])

    def add_ineq_constraint(self, C, d):
        """
        Add the inequality constraint C*x <= d to this polytope. 

        @param C    A matrix of size (m,n)
        @param d    A vector of size (m,)
        """
        m = C.shape[0]
        n = C.shape[1]
        assert n == self.n, "C must be of size (m,n)"
        assert d.shape == (m,), "d must be of size (m,)"

        self.C = np.vstack([self.C,C])
        self.d = np.hstack([self.d,d])

    def check_ineq_redundancy(self, c_prime, d_prime):
        """
        Check whether the inequality constraint 
            c_prime*x <= d_prime 
        is redundant with this polytope. 
        
        @param c_prime    A vector of size (1,n)
        @param d_prime    A vector of size (1,)

        @returns redundnat  A boolean which is True only if the constraint is already enforced. 
        """
        assert c_prime.shape == (1,self.n), "c_prime must be of size (1,n)"
        assert d_prime.shape == (1,), "d_prime must be of size (1,)"

        # We'll check redundancy by solving up a linear program
        #
        #   max  c_prime*x
        #   s.t. C*x <= d
        #        A*x = b
        #        c_prime*x <= d_prime + 1
        #
        # If c_prime*x <= d_prime at optimality, then the constraint is redundant. 
        print("Checking constraint redundancy")

        prog = MathematicalProgram()
        x = prog.NewContinuousVariables(self.n, 'x')

        prog.AddLinearCost(a=-c_prime.T,b=0,vars=x)

        if self.eq_matrices is not None:
            prog.AddConstraint( eq(self.A@x, self.b) )
        if self.ineq_matrices is not None:
            prog.AddConstraint( le(self.C@x, self.d) )

        prog.AddConstraint( le(c_prime@x, d_prime+1) )

        solver = GurobiSolver()
        res = solver.Solve(prog)
        assert res.is_success(), "Infeasible LP in redundancy check"

        if -res.get_optimal_cost() <= d_prime:
            return True
        return False

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
