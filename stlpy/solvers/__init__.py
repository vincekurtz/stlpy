# Test whether various optional dependencies are installed
DRAKE_ENABLED = True
try:
    import pydrake
except ImportError:
    print("WARNING: pydrake import failed, Drake-based solvers disabled.")
    print("         Install drake (https://drake.mit.edu/installation.html)")
    print("         to use the Drake-based solvers.")
    DRAKE_ENABLED = False

SCIPY_ENABLED = True
try:
    import scipy
except ImportError:
    print("WARNING: scipy import failed, scipy-based solvers disabled.")
    print("         Install scipy (https://scipy.org/install/)")
    print("         to use the scipy-based solvers.")
    SCIPY_ENABLED = False

GUROBI_ENABLED = True
try:
    import gurobipy
except ImportError:
    print("WARNING: gurobipy import failed, Gurobi-based solvers disabled.")
    print("         Install Gurobi (https://www.gurobi.com/documentation/)")
    print("         to use the Gurobi-based solvers. Free academic licenses")
    print("         are available.")
    GUROBI_ENABLED = False

# And load the corresponding solvers accordingly
if SCIPY_ENABLED:
    from .scipy.gradient_solver import ScipyGradientSolver

if GUROBI_ENABLED:
    from .gurobi.gurobi_micp import GurobiMICPSolver

if DRAKE_ENABLED:
    from .drake.drake_micp import DrakeMICPSolver
    from .drake.drake_smooth import DrakeSmoothSolver
    from .drake.drake_sos1 import DrakeSos1Solver
