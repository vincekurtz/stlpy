=================================
Solving Control Problems
=================================

Given an STL formula and a dynamical system, the STL synthesis problem
is to discover a system trajectory (inputs, states, outputs) such that
the output signal satisfies the specification. 

This page documents our implementations of several state-of-the-art synthesis
algorithms. They are categorized by the software used to define and solve the
underlying optimization problem. 

Quick summary:

- `DrakeMICPSolver`_: the standard mixed-integer convex programming (MICP) approach. Only linear 
  systems and linear predicates are supported. Finds a globally optimal solution.
  This tends to be the fastest method for short-horizon specifications.
- `DrakeSos1Solver`_: an improved MICP approach which uses fewer binary variables. Tends to
  outperform the standard MICP on long-horizon specifications. Only linear systems and 
  linear predicates are supported. Finds a globally optimal solution.
- `DrakeSmoothSolver`_: optimizes over a smooth approximation of the STL robustness measure. 
  Finds a locally optimal solution. Works with nonlinear systems and predicates.
- `GurobiMICPSolver`_: identical to `DrakeMICPSolver`_, but uses Gurobi's python bindings 
  instead of Drake's.
- `ScipyGradientSolver`_: the simplest (and slowest) method. Optimizes over the 
  (non-smooth) STL robustness measure directly using ``scipy.minimize``. Finds a locally
  optimal solution. Works with nonlinear systems and predicates.

Drake
=====

`Drake <https://drake.mit.edu/>`_ is a modeling, simulation, and control toolbox for robotic
systems. It includes a convienient interface for fomulating and solving optimization problems
and provides bindings to numerous specialized solvers, including Gurobi, Mosek, SNOPT, Ipopt,
and many others.

A complete list of supported solvers and the types of optimization problems they address can
be found `here <https://drake.mit.edu/doxygen_cxx/group__solvers.html>`__.

Installation instructions for Drake can be found `here <https://drake.mit.edu/installation.html>`__.

DrakeMICPSolver
----------------

.. autoclass:: pySTL.solvers.DrakeMICPSolver
    :members: Solve, AddControlBounds, AddQuadraticCost, AddStateBounds
    :show-inheritance:

DrakeSos1Solver
---------------

.. autoclass:: pySTL.solvers.DrakeSos1Solver
    :members: Solve, AddControlBounds, AddQuadraticCost, AddStateBounds
    :show-inheritance:

DrakeSmoothSolver
-----------------

.. autoclass:: pySTL.solvers.DrakeSmoothSolver
    :members: Solve, AddControlBounds, AddQuadraticCost, AddStateBounds, AddRobustnessConstraint
    :show-inheritance:

Gurobi
======

Solvers in this section use the python bindings of `Gurobi <https://www.gurobi.com/>`_, a commercial
optimizer that handles a variety of problem classes, including convex programs, mixed-integer programs,
and non-convex quadratic programs with quadratic constraints. Free academic licenses are available. 

GurobiMICPSolver
----------------

.. autoclass:: pySTL.solvers.GurobiMICPSolver
    :members: Solve, AddControlBounds, AddStateBounds
    :show-inheritance:


Scipy
=====

Solvers in this section are based on the (relatively simple)
`scipy minimize <https://docs.scipy.org/doc/scipy/reference/reference/optimize.minimize-slsqp.html>`_
optimizer, which can be installed with ``pip``:

::

    pip install scipy

Alternative installation instructions can be found `here <https://scipy.org/install/>`_.

ScipyGradientSolver
-------------------

.. autoclass:: pySTL.solvers.ScipyGradientSolver
    :members: Solve, AddQuadraticCost
    :show-inheritance:

Write Your Own Solver
=====================

All the solvers described above inherit from the :class:`.STLSolver` class.
To implement your own optimization-based solver, all you need to do is create 
a new class that inherits from :class:`.STLSolver`.

.. autoclass:: pySTL.solvers.base.STLSolver
    :members:
    :show-inheritance:
