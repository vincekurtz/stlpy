from utils import Polytope

class Partition(Polytope):
    """
    A Partition is a special kind of Polytope which 
    is associated with a list of STLFormulas (really
    state formulas in particular), which hold 
    everywhere within this partition. 
    """
    def __init__(self, polytope, state_formulas):

        # Copy all attributes of polytope to self
        self.__dict__.update(polytope.__dict__)

        # save the state formulas
        self.formulas = state_formulas


