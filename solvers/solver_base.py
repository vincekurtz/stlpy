from abc import ABC, abstractmethod

class STLSolver:
    """
    A simple base class defining a common solver interface
    for different STL synthesis methods. 
    """
    @abstractmethod
    def Solve(self):
        """
        Solve the STL syntheis problem and return a satisfying
        state (x) and input (u) sequence. 
        """
        pass
