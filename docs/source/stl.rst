=====================
Defining STL Formulas
=====================

As described :ref:`here<Signal Temporal Logic>`, STL formulas :math:`\varphi` are defined recursively accoring to the STL syntax:

.. math::

    \varphi = \pi \mid \lnot \varphi \mid \varphi_1 \land \varphi_2 \mid \varphi_1 \lor \varphi_2 \mid 
    G_{[t_1,t_2]}\varphi \mid F_{[t_1,t_2]}\varphi \mid \varphi_1 U_{[t_1,t_2]} \varphi_2

We represent STL formulas :math:`\varphi` using the abstract base class :class:`.STLFormula`. This
base class enables all of the basic STL operations like conjuction (:math:`\land`), disjuction
(:math:`\lor`), always (:math:`G`), until (:math:`U`), and so on. 

Internally, we represent predicates :math:`\pi` using the :class:`.LinearPredicate` class and all
other formulas using the class :class:`.STLTree`. 


.. warning::

    For now, only formulas in positive normal form are supported. That means that negation 
    (:math:`\lnot`) can only be applied to predicates (:class:`.LinearPredicate`). Note that
    any STL formula can be re-written in positive normal form.


STLFormula
==========

.. autoclass:: pySTL.STL.STLFormula
    :members:
    :show-inheritance:

STLTree
=======

.. autoclass:: pySTL.STL.STLTree
    :show-inheritance:

LinearPredicate
===============

.. autoclass:: pySTL.STL.LinearPredicate
    :show-inheritance:

