Tools for testing various formulations of STL synthesis for control of (linear) dynamical systems.

## Dependencies

- python3
- numpy
- scipy (for gradient-based optimization)
- Drake with Gurboi/Mosek (for MICP-based optimization)

## Code structure

The goal is to separate out the definition of an STL formula from the
method used to do synthesis as much as possible. With this in mind:

- `STL` contains the basic definitions of formulas and predicates
- `scenarios` contains examples of constructing STL formulas for common scenarios, such as the reach-avoid problem.
- `solvers` contains various different solvers that solve the same synthesis problem
- Scripts in this folder (like `reach_avoid_example.py`) are examples of defining a specification
    from `scenarios` and using a solver from `solvers` to find a satisfying solution.
