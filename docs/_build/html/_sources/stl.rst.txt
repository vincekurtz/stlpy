=====================
Defining STL Formulas
=====================

As described here, STL formulas :math:`\varphi` are defined recursively accoring to the STL syntax:

.. math::

    \varphi = \pi \mid \lnot \varphi \mid \varphi_1 \land \varphi_2 \mid \varphi_1 \lor \varphi_2 \mid 
    G_{[t_1,t_2]}\varphi \mid F_{[t_1,t_2]}\varphi \mid \varphi_1 U_{[t_1,t_2]} \varphi_2

We represent STL formulas :math:`\varphi` using the abstract base class :class:`.STLFormula`. This
base class enables all of the basic STL operations like conjuction (:math:`\land`), disjuction
(:math:`\lor`), always (:math:`G`), until (:math:`U`), and so on. 

Internally, we represent predicates :math:`\pi` using the :class:`.STLPredicate` class and all
other formulas using the class :class:`.STLTree`. 

STLFormula
==============

.. autoclass:: STL.STLFormula
    :members:
    :show-inheritance:

STLTree
==========

.. autoclass:: STL.STLTree
    :members:
    :show-inheritance:

STLPredicate
============

.. autoclass:: STL.STLPredicate
    :members:
    :show-inheritance:

