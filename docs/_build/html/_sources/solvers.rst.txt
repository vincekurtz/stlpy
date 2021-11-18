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

(for each solver type) List dependencies and link install instructions

Then give class documentation.

    
Scipy Optimize
=========================

.. autoclass:: solvers.ScipyGradientSolver
    :members: Solve
    :show-inheritance:

Gurobi
=========================

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
