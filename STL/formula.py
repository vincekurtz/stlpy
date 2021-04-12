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
        return STLFormula([self,other],"and",[0,0])

    def __and__(self, other):
        """
        Syntatic sugar so we can write `one_and_two = one & two`
        """
        return self.conjunction(other)
    
    def disjunction(self, other):
        """
        Return a new STLFormula which is the disjuction (or) of this
        formula and another one. 

        @param other    The STLFormula or STLPredicate to combine
                        with this one. 

        @returns new    An STLFormula representing (self) and (other)
        """
        return STLFormula([self,other],"or",[0,0])

    def __or__(self, other):
        """
        Syntatic sugar so we can write `one_or_two = one | two`
        """
        return self.disjunction(other)

    def always(self, t1, t2):
        """
        Return a new STLFormula which ensures that this formula holds
        for all of the timesteps between t1 and t2. 

        @param t1   An integer representing the start of the interval
        @param t2   An integer representing the end of the interval

        @returns new    An STLFormula representing G_[t1,t2](self)
        """
        time_interval = [t for t in range(t1,t2+1)]
        subformula_list = [self for t in time_interval]
        return STLFormula(subformula_list,"and",time_interval)

    def eventually(self, t1, t2):
        """
        Return a new STLFormula which ensures that this formula holds
        for at least one timestep between t1 and t2. 

        @param t1   An integer representing the start of the interval
        @param t2   An integer representing the end of the interval

        @returns new    An STLFormula representing F_[t1,t2](self)
        """
        time_interval = [t for t in range(t1,t2+1)]
        subformula_list = [self for t in time_interval]
        return STLFormula(subformula_list,"or",time_interval)

    def until(self, other, t1, t2):
        """
        Return a new STLFormula which ensures that this formula holds
        until some timestep between t1 and t2, at which point the
        other STLFormula (or STLPredicate) holds. 

        @param other    The second STLFormula or STLPredicate that 
                        must hold after this one
        @param t1   An integer representing the start of the interval
        @param t2   An integer representing the end of the interval

        @returns new    An STLFormula representing (self)U_[t1,t2](other)
        """
        # For every candidate swiching time (t_prime), construct a subformula 
        # representing 'self' holding until t_prime, at which point 'other' holds.
        self_until_tprime = []

        for t_prime in range(t1, t2+1):
            time_interval = [t for t in range(t1, t_prime+1)]
            subformula_list = [self for t in range(t1, t_prime)]
            subformula_list.append(other)
            self_until_tprime.append(STLFormula(subformula_list, "and", time_interval))

        # Then we take the disjuction over each of these formulas
        return STLFormula(self_until_tprime, "or", [0 for i in range(len(self_until_tprime))])

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
    def __init__(self, subformula_list, combination_type, timesteps):
        """
        An STL formula is defined by a list of other STLFormulaBase objects
        which are combined together using either conjuction (and) or 
        disjunction (or). 

        @param subformula_list      A list of STLFormulaBase objects (formulas or
                                    predicates) that we'll use to construct this formula. 
        @param combination_type     A string representing the type of operation we'll use 
                                    to combine these objects. Must be either "and" or "or".
        @param timesteps            A list of timesteps that the subformulas must hold at.
                                    This is needed to define the temporal operators.
        """
        # Record the dimension of the signal this formula is defined over
        self.d = subformula_list[0].d

        # Run some type check on the inputs
        assert (combination_type == "and") or (combination_type == "or"), "Invalid combination type"
        assert isinstance(subformula_list, list), "subformula_list must be a list of STLFormula or STLPredicate objects"
        assert isinstance(timesteps, list), "timesteps must be a list of integers"
        assert len(timesteps) == len(subformula_list), "a timestep must be provided for each subformula"
        for formula in subformula_list:
            assert isinstance(formula, STLFormulaBase), "subformula_list must be a list of STLFormula or STLPredicate objects"
            assert formula.d == self.d, "all subformulas must be defined over same dimension of signal"
        for t in timesteps:
            assert isinstance(t, int), "each timestep must be an integer"

        # Simply save the input arguments. We will parse these recursively later on to
        # determine, for example, the formula robustness.
        self.subformula_list = subformula_list
        self.combination_type = combination_type
        self.timesteps = timesteps

    def robustness(self, y, t):
        if self.combination_type == "and":
            return min( [formula.robustness(y,t+self.timesteps[i]) for i, formula in enumerate(self.subformula_list)] )
        else: # combination_type == "or"
            return max( [formula.robustness(y,t+self.timesteps[i]) for i, formula in enumerate(self.subformula_list)] )