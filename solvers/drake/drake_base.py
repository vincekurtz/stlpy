from solvers.base import STLSolver
from pydrake.all import MathematicalProgram

class DrakeSTLSolver(STLSolver):
    """
    A base class for solvers that use the Drake interface to connect with 
    a lower-level solver like Gurobi, Mosek, SNOPT, or IPOPT.
    """
    def __init__(self, spec, sys, x0, T):
        STLSolver.__init__(self, spec, sys, x0, T)
        
        # Create the drake MathematicalProgram instance that will allow
        # us to interface with a MIP solver like Gurobi or Mosek
        self.mp = MathematicalProgram()

        # Flag for whether to solve a convex relaxation of the problem. This is really
        # only useful for MICP-based solvers.
        self.convex_relaxation=False

    def NewBinaryVariables(self, size, name='b'):
        """
        A wrapper for 

            self.mp.NewBinaryVariables()

        that adds continuous variables constrained to [0,1] 
        to the optimization problem if the flag self.convex_relaxation 
        is set to True. 
        """
        if self.convex_relaxation:
            var = self.mp.NewContinuousVariables(size, name)
            for i in range(size):
                self.mp.AddConstraint( 0 <= var[i] )
                self.mp.AddConstraint( var[i] <= 1 )
        else:
            var = self.mp.NewBinaryVariables(size, name)

        return var
    
    def GetVariableData(self):
        """
        Return the number of continuous and binary variables in the current
        (Drake) optimization problem. 
        """
        all_vars = self.mp.decision_variables()
        num_continuous = 0
        num_binary = 0
        for var in all_vars:
            if var.get_type().name == 'CONTINUOUS':
                num_continuous += 1
            elif var.get_type().name == 'BINARY':
                num_binary += 1
            else:
                raise RuntimeError("Unexpected %s-type variable" % var.get_type().name)

        assert self.mp.num_vars() == num_continuous + num_binary

        return num_continuous, num_binary
