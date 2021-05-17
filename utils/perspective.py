## 
#
# Utilities for dealing with perspective functions. 
#
# See Boyd, Convex Optimization Ch. 2.3.3 and  3.2.6
#     Marcucci et al. Shortest Paths in Graphs of Convex Sets, Definition 1
#
##

from utils import Polytope
from pydrake.all import eq, le

def add_perspective_constraint(prog, P, x, lmbda):
    """
    Given a polytope P, add the constraint

        x \in lmbda*P

    to the given Drake MathematicalProgram. 

    @param prog     A Drake MathematicalProgram instance
    @param P        The Polytope
    @param x        An n-dimensional vector of Drake variables or expressions
    @param lmbda    The scaling factor, a Drake variable or expression 
    """
    assert isinstance(P, Polytope), "P must be a polytope"
    assert len(x) == P.n, "Variable size doesn't match polytope dimension"

    if P.has_eq_constraints():
        residual = P.A@x - P.b * lmbda
        prog.AddLinearConstraint(eq( residual, 0 ))

    if P.has_ineq_constraints():
        residual = P.C@x - P.d * lmbda
        prog.AddLinearConstraint(le( residual, 0 ))

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

