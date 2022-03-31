==========
Benchmarks
==========

Benchmark scenarios for evaluating 
different control approaches. 

Simple Scenarios
====================

Reach-Avoid
-----------

A robot needs to avoid and obstacle and reach a goal.

.. autofunction::
    pySTL.benchmarks.reach_avoid.reach_avoid_specification
    

Either-Or
---------

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

Helper functions
==========================
