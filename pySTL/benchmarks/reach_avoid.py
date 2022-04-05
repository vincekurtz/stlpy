from .base import BenchmarkScenario
from .common import (inside_rectangle_formula,
                     outside_rectangle_formula,
                     make_rectangle_patch)
from ..systems import DoubleIntegrator

class ReachAvoid(BenchmarkScenario):
    r"""
    A 2D mobile robot with double integrator dynamics must
    avoid an obstacle (:math:`\mathcal{O}`) before reaching a goal (:math:`\mathcal{G}`):

    .. math::

        \varphi = G_{[0,T]} \lnot \mathcal{O} \land F_{[0,T]} \mathcal{G}

    :param goal_bounds:      a tuple ``(xmin, xmax, ymin, ymax)`` defining a
                             rectangular goal region.
    :param obstacle_bounds:  a tuple ``(xmin, xmax, ymin, ymax)`` defining a
                             rectangular obstacle.
    :param T:                the time horizon for this scenario.
    """
    def __init__(self, goal_bounds, obstacle_bounds, T):
        self.goal_bounds = goal_bounds
        self.obstacle_bounds = obstacle_bounds
        self.T = T

    def GetSpecification(self):
        # Goal Reaching
        at_goal = inside_rectangle_formula(self.goal_bounds, 0, 1, 6)

        # Obstacle Avoidance
        not_at_obstacle = outside_rectangle_formula(self.obstacle_bounds, 0, 1, 6)

        # Put all of the constraints together in one specification
        spec = not_at_obstacle.always(0, self.T) & at_goal.eventually(0, self.T)

        return spec

    def GetSystem(self):
        sys = DoubleIntegrator(2)
        return sys

    def add_to_plot(self, ax):
        # Make and add rectangular patches
        obstacle = make_rectangle_patch(*self.obstacle_bounds, color='k', alpha=0.5)
        goal = make_rectangle_patch(*self.goal_bounds, color='green', alpha=0.5)
        ax.add_patch(obstacle)
        ax.add_patch(goal)

        # set the field of view
        ax.set_xlim((0,10))
        ax.set_ylim((0,10))
        ax.set_aspect('equal')
