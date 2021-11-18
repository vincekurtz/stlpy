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
    :members: Solve
    :show-inheritance:

.. autoclass:: solvers.GurobiLCPSolver
    :members: Solve
    :show-inheritance:

Drake
=============================

`Drake <https://drake.mit.edu/>`_ is a modeling, simulation, and control toolbox for robotic
systems. It includes a convienient interface for fomulating and solving optimization problems
and provides bindings to numberous specialized solvers, including Gurobi, Mosek, SNOPT, Ipopt,
and many others.

A complete list of supported solvers and the types of optimization problems they address can
be found `here <https://drake.mit.edu/doxygen_cxx/group__solvers.html>`__.

Installation instructions for Drake can be found `here <https://drake.mit.edu/installation.html>`__.

.. autoclass:: solvers.DrakeMICPSolver
    :members: Solve
    :show-inheritance:

.. autoclass:: solvers.DrakeLCPSolver
    :members: Solve
    :show-inheritance:

Knitro
=============================

Solvers in this section use the commercial `Artelys Knitro <https://www.artelys.com/solvers/knitro/>`_ solver. 
Knitro is a commercial solver which aims to find high-quality locally optimal solutions to nonconvex
problems. It tends to perform particularly well on problems with complementarity constraints. 

More information on how to install Knitro, the algorithms they use, and how to obtain a license
can be found `here <https://www.artelys.com/docs/knitro/1_introduction.html>`__.

.. autoclass:: solvers.KnitroLCPSolver
    :members: Solve
    :show-inheritance:

Write Your Own Solver
=====================

All the solvers described above inherit from the :class:`.STLSolver` class.
To implement your own optimization-based solver, all you need to do is create 
a new class that inherits from :class:`.STLSolver`.

.. autoclass:: solvers.STLSolver
    :members:
    :show-inheritance:
