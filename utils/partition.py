from utils import Polytope

class Partition:
    """
    A Partition consists of a (bounded) polytope and a list
    of STLFormulas that hold everywhere in this polytope
    """
    def __init__(self, polytope, formulas):
        self.polytope = polytope
        self.formulas = formulas

    def plot(self, **kwargs):
        """
        Make a (2d) plot of this partition.
        """
        self.polytope.plot_2d(**kwargs)

    def __str__(self):
        string = "%s-d Partition with: " % self.polytope.n
        for f in self.formulas:
            string += "\n    %s" % f
        return string

