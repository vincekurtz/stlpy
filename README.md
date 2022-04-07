# STLpy

A python library for control from Signal Temporal Logic (STL) specifications.

## Installation

Via pip: *coming soon*

From source:
```
git clone https://github.com/vincekurtz/stlpy
cd stlpy
python setup.py install
```

## Dependencies

### Required Dependencies

These are needed to use the library's basic functionality.

- python version 3.6 or higher
- numpy
- treelib
- matplotlib

### Optional Dependencies

These are optional, and enable specific solvers.

- scipy (for gradient-based optimization)
- Drake with Gurboi/Mosek enabled (for MICP-based optimization)
- Drake with SNOPT enabled (for smooth optimization with sparse SQP)
- Gurobi python bindings (version 9.0 or higher) for MICP-based optimization

## Usage

See the `examples` folder and the documentation.

