========
About
========

**stlpy** is a python library for control from Signal Temporal Logic (STL) specifications. 

This software is designed with the following goals in mind:

    - Provide a :ref:`simple python interface<Defining STL Formulas>` for 
      dealing with STL formulas
    - Provide :ref:`high-quality implementations<Solving Control Problems>` 
      of several state-of-the-art synthesis algorithms, including Mixed-Integer
      Convex Programming (MICP) and gradient-based optimization.
    - Make it easy to design and evaluate 
      :ref:`new synthesis algorithms<Write Your Own Solver>`. 
    - Provide a variety of :ref:`benchmark scenarios<Benchmarks>`
      that can be used to test new algorithms.

If this software is helpful to you, if you have a new synthesis method you would like to see
included in the package, or if you have any questions, feel free to `reach out`_. 

.. _reach out: vjkurtz@gmail.com

Signal Temporal Logic
=====================

Signal Temporal Logic (STL) is a formal language that can be used to define 
complex control objectives for robotic and cyber-physical systems. STL is similar to 
boolean logic in that it includes boolean operators like "and" (:math:`\land`), "or"
(:math:`\lor`) and "not" (:math:`\lnot`), but it also includes some special temporal
operators like "always/globally" (:math:`G_{[t_1,t_2]}`), "eventually/finally" 
(:math:`F_{[t_1,t_2]}`), and "until" (:math:`U_{[t_1,t_2]}`).

As the name suggests, STL is defined over continuous-valued signals:

.. math::

    y = y_0,y_1,y_2,\dots,y_T,

where :math:`y_t \in \mathbb{R}^p` is the value of the signal at timestep :math:`t`.

.. note::

    This software focuses on discrete-time signals of finite length, though STL is also
    well-defined for infinitely long and continuous-time signals. 

The fundamental building blocks of STL are *predicates* :math:`\pi`, defined by inequalities

.. math::

    \pi = ( g^{\pi}(y) > 0 )

Many (though not all) STL synthesis algorithms require linear predicates, i.e., 
:math:`g^{\pi}(y) = a y - b`.

STL Syntax
----------

The *syntax* of STL defines what counts as a valid STL formula. In short form, STL's synatax is

.. math::

    \varphi = \pi \mid \lnot \varphi \mid \bigvee_i \varphi_i \mid \bigwedge_i \varphi_i
    \mid G_{[t_1,t_2]} \varphi \mid F_{[t_1,t_2]} \varphi \mid \varphi_1 U_{[t_1,t_2]} \varphi_2

This essentially means that:

- All predicates :math:`\pi` are valid STL formulas.
- Given an STL formula :math:`\varphi`, its negation :math:`\lnot \varphi` 
  is also a valid STL formula.
- The conjunction ("and", :math:`\land`) of multiple STL formulas is a valid STL formula.
- The disjunction ("or", :math:`\lor`) of multiple STL formulas is a valid STL formula.
- Given an STL formula :math:`\varphi`, :math:`G_{[t_1,t_2]} \varphi` (always :math:`\varphi`)
  is also a valid STL formula.
- Given an STL formula :math:`\varphi`, :math:`F_{[t_1,t_2]} \varphi` (eventually :math:`\varphi`)
  is also a valid STL formula.
- Given two STL formulas :math:`\varphi_1` and :math:`\varphi_2`, :math:`\varphi_1 U_{[t_1,t_2]} \varphi_2`
  (:math:`\varphi_1` until :math:`\varphi_2`) is also a valid STL formula.

STL Semantics
-------------

As with any formal language, while the syntax defines what counts as a formula, 
the *semantics* defines what a formula means. Here we'll present a fairly inutitive
explanation of the semantics:

- Saying that a signal satisfies a specification (:math:`y \vDash \varphi`) is equivalent
  to saying that the first timestep of the signal satisfies the specification.
- Predicates: :math:`\pi` holds if the inequality :math:`g^{\pi}(y) > 0` holds.
- Conjunction: :math:`\varphi_1 \land \varphi_2` holds if both subformulas hold. 
- Disjunction: :math:`\varphi_1 \lor \varphi_2` holds if at least 
  one of the subformulas holds.
- Always: :math:`G_{[t_1,t_2]} \varphi` holds if :math:`\varphi` holds at all timesteps between 
  :math:`t_1` and :math:`t_2`.
- Eventually: :math:`F_{[t_1,t_2]} \varphi` holds if :math:`\varphi` holds at at least one
  timestep between :math:`t_1` and :math:`t_2`.
- Until: :math:`\varphi_1 U_{[t_1,t_2]} \varphi_2` holds if :math:`\varphi_2` holds at some
  timestep between :math:`t_1` and :math:`t_2`, and :math:`\varphi_1` holds before that point.

.. note::

    A more rigorous recursive definition of the STL semantics 
    can be found in most STL-related papers. 

STL is also equipped with a "robustness measure" :math:`\rho^{\varphi}(y)`. This is a function
that maps a signal to a scalar value. :math:`\rho` is positive only if the signal satisfies the 
specification, and negative only if the signal does not satisfy the specification. The robustness
measure allows us to consider "more satisfying" signals and "closer-to-satisfying" signals, 
which is very useful for optimization.

This software provides tools for :ref:`working with STL formulas<Defining STL Formulas>`,
including computation of the robustness measure. See :ref:`this example<A Simple Example>`
for more details.

STL for Control
---------------

STL becomes particularly useful when we consider the signal :math:`y` to be the output
of a :ref:`dynamical system<Modeling Control Systems>`:

.. math::

    x_{t+1} = f(x_t, u_t),

    y_t = g(x_t, u_t),

where :math:`x_t` is the system state and :math:`u_t` is a control input. 

In this context, STL specifications formally define some desired behavior for the system.
Such specifications can become quite complicated (see the :ref:`Benchmarks` for
some examples).

One thing we might like to do is find a sequence of control inputs such that the resulting
output trajectory satisfies the specification: this is the *trajectory synthesis* problem. 

STL trajectory synthesis is an NP-hard problem, and efficient algorithms remain an area
of active research. This software provides high-quality 
:ref:`implementations<Solving Control Problems>` of several 
state-of-the-art algorithms as well as an :ref:`interface<Write Your Own Solver>`
for designing new algorithms.

Citing stlpy
============

To reference **stlpy** in academic research, 
please cite `our paper <https://arxiv.org/abs/2204.06367>`_:

.. code-block:: text

    @article{kurtz2022mixed,
      title={Mixed-Integer Programming for Signal Temporal Logic with Fewer Binary Variables},
      author={Kurtz, Vince and Lin, Hai},
      journal={arXiv preprint arXiv:2204.06367},
      year={2022}
    }

References for specific synthesis methods can be found in the 
:ref:`solver documentation <Solving Control Problems>`.
