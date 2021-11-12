from solvers.solver_base import STLSolver
from pydrake.all import MathematicalProgram

class DrakeSTLSolver(STLSolver):
    """
    A base class for solvers that

        1) Use the Drake interface to connect with a lower-level solver,
            like Gurobi or Mosek
        2) Consider a running cost of the form x'Qx + u'Ru

    """
    def __init__(self, spec, A, B, Q, R, x0, T):
        """
        Initialize the solver.

        @param spec     An STLFormula describing the specification
        @param A        An (n,n) numpy matrix describing state dynamics
        @param B        A (n,m) numpy matrix describing control input dynamics
        @param Q        A (n,n) matrix specifing a running state cost
        @param R        A (m,m) matrix specifing a running control cost
        @param x0       The initial state of the system.
        @param T        An integer specifiying the number of timesteps.
        """
        STLSolver.__init__(self, spec, A, B, x0, T)
        self._include_quadratic_cost(Q,R)
        
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


