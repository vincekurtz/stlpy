## 
#
# Utilities for dealing with perspective functions. 
#
# See Boyd, Convex Optimization Ch. 2.3.3 and  3.2.6
#     Marcucci et al. Shortest Paths in Graphs of Convex Sets, Definition 1
#
##

from pydrake.all import eq

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
    prog.AddRotatedLorentzConeConstraint(lmbda, l, x.T@Q@x)

def add_LCQ_perspective_cost(prog, Q, A, b, x, lmbda):
    """
    Add a linear-constrained quadratic (LCQ) perspective cost to the given
    given Drake MathematicalProgram, where f(x) is a quadratic function

        f(x) = x'Qx

    and the optimization is subject to the (perspective of) the linear constraint

        Ax = b.

    This results in the perspective function taking values

        f_tilde(lmbda, x) = {  x'Qx/lmbda    if lmbda > 0 and Ax = b
                            {  0             if lmbda = 0 and x'Qx = 0 and Ax = b
                            {  infinity      otherwise.

    This can be added to the optimization problem as an SOCP as follows:

        min  l
        s.t. lmbda*l >= x'Qx
             lmbda >= 0
             l >= 0
             Ax = b*lmbda

    where the last constraint comes from the fact that the perspective of 
        
        Ax - b

    is given by 

        Ax - b*lmbda.

    (See Marcucci et al. Appendix C)
    """
    add_quadratic_perspective_cost(prog, Q, x, lmbda)
    prog.AddLinearConstraint(eq( A@x - b*lmbda, 0 ))

if __name__=="__main__":
    # Testing
    from pydrake.all import MathematicalProgram, GurobiSolver
    import numpy as np

    prog = MathematicalProgram()

    Q = np.eye(2)
    A = np.eye(2)
    b = 2*np.ones((2,))
    x = prog.NewContinuousVariables(2,'x')
    lmbda = prog.NewContinuousVariables(1,'lambda')[0]

    add_linear_constrained_quadratic_perspective_cost(prog, Q, A, b, x, lmbda) 
    prog.AddConstraint(lmbda == 1)

    solver = GurobiSolver()
    res = solver.Solve(prog)

    print(res.is_success())

    print(res.GetSolution(x))
    print(res.get_optimal_cost())
    print(res.GetSolution(lmbda))


