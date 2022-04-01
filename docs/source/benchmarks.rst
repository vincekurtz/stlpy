==========
Benchmarks
==========

Benchmark scenarios for evaluating 
different control approaches. 

Simple Scenarios
====================

Reach-Avoid
-----------

.. autoclass::
    pySTL.benchmarks.ReachAvoid
    :members:
    :show-inheritance:

Either-Or
---------

.. autoclass::
    pySTL.benchmarks.EitherOr
    :members:
    :show-inheritance:

In addition to avoiding an obstacle and reaching a goal, the robot
must visit one of two intermediate target regions. 

Narrow Passage
--------------

A robot must avoid several obstacles and reach one of two
goals, and the passageway between obstacles is narrow.

Complex/Scalable Scenarios
==========================

Multi-Target
------------

A robot must avoid many obstacles and visit at least one target of various types.

Key-Door
--------

The robot must visit several target regions (rooms), but before entering each room it must
visit a different location to pick up a key. 

.. autoclass::
    pySTL.benchmarks.DoorPuzzle
    :members:
    :show-inheritance:

Adding New Benchmarks
=======================

To add additional benchmark scenarios, simply create
an object that inherits from the following ``BenchmarkScenario`` class:

.. autoclass::
    pySTL.benchmarks.base.BenchmarkScenario
    :members:
    :show-inheritance:

Helper functions
==========================

inside_rectangle_formula
------------------------
.. autofunction::
    pySTL.benchmarks.common.inside_rectangle_formula

outside_rectangle_formula
-------------------------
.. autofunction::
    pySTL.benchmarks.common.outside_rectangle_formula

make_rectangle_patch
--------------------
.. autofunction::
    pySTL.benchmarks.common.make_rectangle_patch
