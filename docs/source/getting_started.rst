=================================
Getting Started
=================================

Installation
=================================

Via pip: *coming soon*

From source:
::

    git clone https://github.com/vincekurtz/pySTL
    cd pySTL
    python setup.py install


Dependencies
=================================

Required Packages
-----------------

The following are required for the core functionality
of this software (defining STL formulas). 

- Python version 3.6 or higher
- numpy
- treelib
- matplotlib

Solver-Specific Packages
------------------------

These are solver-specific dependencies, and are needed to enable 
particular :ref:`synthesis algorithms<Solving Control Problems>`.
STL formulas can be defined and evaluated without these dependencies.

- `scipy <https://scipy.org/>`_ - used for the most basic gradient-based optimization
- `Gurobi <https://gurobi.com/>`_ version 9.1 or higher (with python bindings) - used 
  for basic mixed-integer programming
- `Drake <https://drake.mit.edu/>`_ with Gurobi/Mosek enabled - used 
  for more advanced mixed-integer programming
- Drake with SNOPT enabled - used for more efficient gradient-based optimization

.. note::
    
    Drake with Gurobi/Mosek enabled requires compilation from source. See
    `here <https://drake.mit.edu/from_source.html#building-the-python-bindings>`_
    for more details.

.. note::

    Drake's binary releases include SNOPT (no license required). 

A Simple Example
=================================

*coming soon*

More Examples
=================================

Can be found in the `examples` folder.
