from abc import ABC, abstractmethod

class BenchmarkScenario(ABC):
    """
    An abstract base class defining a benchmark
    scenario for STL synthesis.
    """
    @abstractmethod
    def GetSpecification(self):
        """
        Get the STL specification associated with this scenario.

        :return spec: an ``STLFormula`` describing the specification.
        """
        pass
   
    @abstractmethod
    def GetSystem(self):
        """
        Get the system dynamics model associated with this
        scenario. 

        :return sys: a ``LinearSystem`` or ``NonlinearSystem`` 
                     specifying the system dynamics.
        """
        pass

    @abstractmethod
    def add_to_plot(self, ax):
        """
        Add a visualization of this specification
        to the given ``matplotlib`` axis.

        :param ax:  The ``matplotlib`` axis object to add the 
                    visualization to.
        """
        pass
