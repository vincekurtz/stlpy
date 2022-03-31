Tools for testing various formulations of STL synthesis for control of (linear) dynamical systems.

## Dependencies

### Required Dependencies

These are needed to use the library's basic functionality.

- python3
- numpy
- matplotlib

### Optional Dependencies

These are optional, and enable specific solvers.

- scipy (for gradient-based optimization)
- Drake with Gurboi/Mosek enabled (for MICP-based optimization)
- Drake with SNOPT enabled (for smooth optimization with sparse SQP)
- Gurobi python bindings (v9 or greater) for MICP-based optimization

## Installation

```
pip install .
```

## Usage

See the `examples` folder and the documentation.

