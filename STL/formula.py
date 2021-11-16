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

    @abstractmethod
    def is_state_formula(self):
        """
        Indicate whether this formula is a state formula, e.g.,
        a predicate or the result of boolean operations over 
        predicates. 

        @returns status     A boolean, True only if self is a state formula.
        """
        pass
    
    @abstractmethod
    def is_disjunctive_state_formula(self):
        """
        Indicate whether this formula is a state formula defined by
        only disjunctions (or) over predicates. 

        @returns status     A boolean, True only if self is a disjunctive state formula.
        """
        pass

    @abstractmethod
    def is_conjunctive_state_formula(self):
        """
        Indicate whether this formula is a state formula, defined by
        only conjunctions (and) over predicates.

        @returns status     A boolean, True only if self is a conjunctive state formula.
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
        Return a new STLFormula which is the disjunction (or) of this
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
        formula = STLFormula(subformula_list, "and", time_interval) 
        if self.name is not None:
            formula.name = "always [%s,%s] %s" % (t1,t2,self.name)
        return formula

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
        formula = STLFormula(subformula_list, "or", time_interval) 
        if self.name is not None:
            formula.name = "eventually [%s,%s] %s" % (t1,t2,self.name)
        return formula

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

        # Then we take the disjunction over each of these formulas
        return STLFormula(self_until_tprime, "or", [0 for i in range(len(self_until_tprime))])

class STLFormula(STLFormulaBase):
    """
    An STL formula (in positive normal form) defined by 
    the following operations on STLPredicates and other STLFormulas:

        - conjunction (and)
        - disjunction (or)
        - always (globally)
        - eventually (finally)
        - until

    """
    def __init__(self, subformula_list, combination_type, timesteps, name=None):
        """
        An STL formula is defined by a list of other STLFormulaBase objects
        which are combined together using either conjunction (and) or 
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

        # Save the given name for pretty printing
        self.name=name

    def robustness(self, y, t):
        if self.combination_type == "and":
            return min( [formula.robustness(y,t+self.timesteps[i]) for i, formula in enumerate(self.subformula_list)] )
        else: # combination_type == "or"
            return max( [formula.robustness(y,t+self.timesteps[i]) for i, formula in enumerate(self.subformula_list)] )

    def is_state_formula(self):
        boolean_operation = all([self.timesteps[i] == self.timesteps[0] for i in range(len(self.timesteps))])
        children_are_state_formulas = all([subformula.is_state_formula() for subformula in self.subformula_list])

        return boolean_operation and children_are_state_formulas

    def is_disjunctive_state_formula(self):
        boolean_operation = all([self.timesteps[i] == self.timesteps[0] for i in range(len(self.timesteps))])
        children_match = all([s.is_disjunctive_state_formula() for s in self.subformula_list])

        return boolean_operation and children_match and self.combination_type == "or"

    def is_conjunctive_state_formula(self):
        boolean_operation = all([self.timesteps[i] == self.timesteps[0] for i in range(len(self.timesteps))])
        children_match = all([s.is_conjunctive_state_formula() for s in self.subformula_list])

        return boolean_operation and children_match and self.combination_type == "and"

    def __str__(self):
        if self.name is None:
            return "{ Formula of %s-type, %s subformulas }" % (self.combination_type, len(self.subformula_list))
        else:
            return "{ Formula " + self.name + " }"
