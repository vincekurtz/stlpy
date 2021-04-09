import numpy as np
from abc import ABC, abstractmethod

class STLFormulaBase(ABC):
    """
    An abstract class which describes an STL formula
    (i.e. a specification made up of logical operations 
    over predicates) or an STL predicate (i.e. the 
    simplest possible formula).
    """

    @abstractmethod
    def robustness(self, y, t):
        """
        Return the robustness measure of this formula for the
        given signal y[t], evaluated at timestep t. 

        @param y    A dxT numpy array representing the signal
                    to evaluate, where d is the dimension of
                    the signal and T is the number of timesteps
        @param t    The timestep to evaluate the signal at. This 
                    is typically 0 for the full formula. 

        @returns rho    The robustness measure, which is positive only
                        if the signal satisfies the specification
        """
        pass

    def conjunction(self, other):
        """
        Return a new STLFormula which is the conjunction (and) of this
        formula and another one. 

        @param self     This STLFormula or STLPredicate
        @param other    The STLFormula or STLPredicate to combine
                        with this one. 

        @returns new    An STLFormula representing (self) and (other)
        """
        pass

class STLPredicate(STLFormulaBase):
    """
    A (linear) stl predicate defined by

        A*y_t - b >= 0

    where y_t is the value of the signal 
    at a given timestep t.
    """
    def __init__(self, A, b):
        # Convert provided constraints to numpy arrays
        self.A = np.asarray(A)
        self.b = np.atleast_1d(b)
        
        # Some dimension-related sanity checks
        assert (self.A.shape[0] == 1), "A must be of shape (1,d)"
        assert (self.b.shape == (1,)), "b must be of shape (1,)"
        
        # Store the dimensionality of y_t
        self.d = self.A.shape[1]

    def robustness(self, y, t):
        assert isinstance(y, np.ndarray), "y must be a numpy array"
        assert isinstance(t, int), "timestep t must be an integer"
        assert y.shape[0] == self.d, "y must be of shape (d,T)"
        assert y.shape[1] > t, "requested timestep %s, but y only has %s timesteps" % (t, y.shape[1])

        return self.A@y[:,t] - self.b

class STLFormula(STLFormulaBase):
    """
    An STL formula (in positive normal form) defined by 
    the following operations on STLPredicates and other STLFormulas:

        - conjunction (and)
        - disjuction (or)
        - always (globally)
        - eventually (finally)
        - until

    """
    def __init__(self, subformula_list, combination_type):
        """
        An STL formula is defined by a list of other STLFormulaBase objects
        which are combined together using either conjuction (and) or 
        disjunction (or). 

        @param subformula_list      A list of STLFormulaBase objects (formulas or
                                    predicates) that we'll use to construct this formula. 
        @param combination_type     A string representing the type of operation we'll use 
                                    to combine these objects. Must be either "and" or "or".
        """
        # Run some type check on the inputs
        assert (combination_type == "and") or (combination_type == "or"), "Invalid combination type"
        assert isinstance(subformula_list, list), "subformula_list must be a list of STLFormula or STLPredicate objects"
        for formula in subformula_list:
            assert isinstance(formula, STLFormulaBase), "subformula_list must be a list of STLFormula or STLPredicate objects"

        # Simply save the subformula list and combination type. That will let
        # us parse these recursively later on
        self.subformula_list = subformula_list
        self.combination_type = combination_type

    def robustness(self, y, t):
        if self.combination_type == "and":
            return min( [formula.robustness(y,t) for formula in self.subformula_list] )
        else: # combination_type == "or"
            return max( [formula.robustness(y,t) for formula in self.subformula_list] )

if __name__=="__main__":
    pi0 = STLPredicate([[0,1]],[1])  # y[0] > 1
    pi1 = STLPredicate([[1,0]],[1])  # y[1] > 1

    y = np.array([[0,0],[2,0],[0,2],[2,2]]).T

    phi = STLFormula([pi0,pi1],"or")

    for t in range(4):
        print(t)
        print(pi0.robustness(y,t))
        print(pi1.robustness(y,t))
        print(phi.robustness(y,t))
        print("")


