from utils import Polytope

class Partition:
    """
    A Partition consists of a (bounded) polytope and a list
    of STLPredicate that hold everywhere in this polytope
    """
    def __init__(self, polytope, predicates):
        self.polytope = polytope
        self.predicates = predicates

    def plot(self, **kwargs):
        """
        Make a (2d) plot of this partition.
        """
        self.polytope.plot_2d(**kwargs)

    def __str__(self):
        string = "%s-d Partition with: " % self.polytope.n
        for pred in self.predicates:
            string += "\n    %s" % pred
        return string

