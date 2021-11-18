=================================
Solving Control Problems
=================================

Overview and approach

List (with links) of different solver approaches:

    - Mixed-Integer Programming
        - Gurobi
        - Drake (Gurobi and Mosek)
    - Smooth Local Optimization
        - Scipy
        - Drake (SNOPT and IPOPT)
    - Complementarity Problems
        - Gurobi
    - Non-smooth Local Optimization
        - Drake (SNOPT and IPOPT)
        - Knitro

Scipy Optimize
=========================

Solvers in this section are based on the (relatively simple)
`scipy minimize <https://docs.scipy.org/doc/scipy/reference/reference/optimize.minimize-slsqp.html>`_
optimizer, which can be installed with ``pip``:

::

    pip install scipy

Alternative installation instructions can be found `here <https://scipy.org/install/>`_

.. autoclass:: solvers.ScipyGradientSolver
    :members: Solve
    :show-inheritance:

Gurobi
=========================

Solvers in this section use the python bindings of `Gurobi <https://www.gurobi.com/>`_, a commercial
optimizer that handles a variety of problem classes, including convex programs, mixed-integer programs,
and non-convex quadratic programs with quadratic constraints. Free academic licenses are available. 

.. autoclass:: solvers.GurobiMICPSolver
    :members: Solve, _encode_min, _encode_max
    :show-inheritance:

Drake
=============================

Knitro
=============================

Write Your Own Solver
=====================

All the solvers described above inherit from the :class:`.STLSolver` class.
To implement your own optimization-based solver, all you need to do is create 
a new class that inherits from :class:`.STLSolver`.

.. autoclass:: solvers.STLSolver
    :members:
    :show-inheritance:
