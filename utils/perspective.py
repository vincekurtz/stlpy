## 
#
# Utilities for dealing with perspective functions. 
#
# See Boyd, Convex Optimization Ch. 2.3.3 and  3.2.6
#     Marcucci et al. Shortest Paths in Graphs of Convex Sets, Definition 1
#
##

def add_quadratic_perspective_cost(prog, Q, x, lmbda):
    """
    Add the (extended) perspective function f_tilde(lmbda,x) as a cost to the
    given Drake MathematicalProgram, where f(x) is a quadratic function

        f(x) = x'Qx

    and the perspective is defined as 

        f_tilde(lmbda, x) = {  x'Qx/lmbda    if lmbda > 0,
                            {  0             if lmbda = 0 and x'Qx = 0,
                            {  infinity      otherwise.

    This can be added to the optimization problem as an SOCP as follows:

        min  l
        s.t. lmbda*l >= x'Qx
             lmbda >= 0
             l >= 0
    """
    l = prog.NewContinuousVariables(1,'l')[0]
    prog.AddCost(l)
    return prog.AddRotatedLorentzConeConstraint(lmbda, l, x.T@Q@x)

def add_linear_constrained_perspective_cost(prog, Q, A, b, lmbda):
    pass

if __name__=="__main__":
    from pydrake.all import MathematicalProgram, GurobiSolver
    import numpy as np

    prog = MathematicalProgram()

    Q = np.eye(2)
    x = prog.NewContinuousVariables(2,'x')
    lmbda = prog.NewContinuousVariables(1,'lambda')[0]

    add_quadratic_perspective_cost(prog, Q, x, lmbda)

    prog.AddConstraint( x[0] + x[1] == 2 )
    prog.AddConstraint( lmbda == 0 )

    solver = GurobiSolver()
    res = solver.Solve(prog)

    print(res.is_success())

    print(res.GetSolution(x))
    print(res.get_optimal_cost())
    print(res.GetSolution(lmbda))


