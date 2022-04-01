==========
Benchmarks
==========

Benchmark scenarios for evaluating 
different control approaches. 

Reach-Avoid
===========

A robot must avoid an obstacle and reach a goal.

.. autoclass::
    pySTL.benchmarks.ReachAvoid
    :members:
    :show-inheritance:

Either-Or
=========

In addition to avoiding an obstacle and reaching a goal, the robot
must visit one of two intermediate target regions and stay there
for several timesteps.

.. autoclass::
    pySTL.benchmarks.EitherOr
    :members:
    :show-inheritance:

Narrow Passage
==============

A robot must avoid several obstacles and reach one of two
goals, and the passageway between obstacles is narrow.

.. autoclass::
    pySTL.benchmarks.NarrowPassage
    :members:
    :show-inheritance:

Multi-Target
============

A robot must avoid many obstacles and visit at least one target of various types.

.. autoclass::
    pySTL.benchmarks.RandomMultitarget
    :members:
    :show-inheritance:

Key-Door
========

The robot must visit several target regions (rooms), but before entering each room it must
visit a different location to pick up a key. 

.. autoclass::
    pySTL.benchmarks.DoorPuzzle
    :members:
    :show-inheritance:

Stepping-Stones
===============

The robot can only step in certain areas before reaching a goal.

.. autoclass::
    pySTL.benchmarks.SteppingStones
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
