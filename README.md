[![Documentation Status](https://readthedocs.org/projects/stlpy/badge/?version=latest)](http://stlpy.readthedocs.io/?badge=latest)
[![PyPi version](https://badgen.net/pypi/v/stlpy/)](https://pypi.org/project/stlpy)
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

# stlpy

A python library for control from Signal Temporal Logic (STL) specifications. 

Includes implementations of several state-of-the-art [synthesis algorithms](https://stlpy.readthedocs.io/en/latest/solvers.html) and [benchmark specifications](https://stlpy.readthedocs.io/en/latest/benchmarks.html) (shown below). 

| <img src="docs/source/images/either_or.png" alt="drawing" width="200"/> |<img src="docs/source/images/nonlinear_reach_avoid.png" alt="drawing" width="200"/> | <img src="docs/source/images/door_puzzle.png" alt="drawing" width="200"/> |<img src="docs/source/images/stepping_stones.png" alt="drawing" width="200"/> |
| --- | --- | --- | --- | 

### Documentation

Can be found online at [stlpy.readthedocs.io](https://stlpy.readthedocs.io/en/latest/index.html).

### Installation

```
pip install stlpy
```

The basic installation allows for defining and evaluating STL formulas. The more advanced solvers 
require one or more of the following additional packages. See the 
[documentation](https://stlpy.readthedocs.io/en/latest/solvers.html)
for more details.

- [scipy](https://scipy.org/install/) (for gradient-based optimization)
- [Drake with Gurboi/Mosek](https://drake.mit.edu/from_source.html#building-the-python-bindings) enabled (for MICP-based optimization)
- [Drake with SNOPT](https://drake.mit.edu/from_binary.html#binary-packages) enabled (for smooth optimization with sparse SQP)
- [Gurobi](https://www.gurobi.com/documentation/9.5/quickstart_linux/cs_using_pip_to_install_gr.html) python bindings (version 9.0 or higher) for MICP-based optimization

### Usage

See the [examples](examples) and the [documentation](https://stlpy.readthedocs.io/en/latest/getting_started.html#a-simple-example).

### Contributing

If you have a new STL trajectory synthesis algorithm or benchmark scenario you would like to see included in this package, please open a [pull request](https://github.com/vincekurtz/stlpy/pulls). 

### Citing stlpy

To reference **stlpy** in academic research, please cite [our paper](https://arxiv.org/abs/2204.06367):

```
@article{kurtz2022mixed,
  title={Mixed-Integer Programming for Signal Temporal Logic with Fewer Binary Variables},
  author={Kurtz, Vince and Lin, Hai},
  journal={arXiv preprint arXiv:2204.06367},
  year={2022}
}
```

References for specific synthesis methods can be found in the [solver documentation](https://stlpy.readthedocs.io/en/latest/solvers.html).
