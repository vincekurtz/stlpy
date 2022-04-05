from .base import BenchmarkScenario
from .common import (inside_rectangle_formula,
                     outside_rectangle_formula,
                     make_rectangle_patch)
from ..systems import DoubleIntegrator

class NarrowPassage(BenchmarkScenario):
    r"""
    A 2D mobile robot with double integrator dynamics must navigate around
    several obstacles (:math:`\mathcal{O}_i`) before reaching one of two
    goals (:math:`\mathcal{G}_i`).

    .. math::

        \varphi = F_{[0,T]}(\mathcal{G}_1 \lor \mathcal{G}_2) \land
            G_{[0,T]} \left( \bigwedge_{i=1}^4 \lnot \mathcal{O}_i \right)

    :param T:   The time horizon of the specification.
    """
    def __init__(self, T):
        # Define obstacle and goal regions by their bounds,
        # (xmin, xmax, ymin, ymax)
        self.obstacles = [(2,5,4,6),
                          (5.5,9,3.8,5.7),
                          (4.6,8,0.5,3.5),
                          (2.2,4.4,6.4,11)]
        self.goals = [(7,8,8,9),
                      (9.5,10.5,1.5,2.5)]
        self.T = T

    def GetSpecification(self):
        # Goal Reaching
        goal_formulas = []
        for goal in self.goals:
            goal_formulas.append(inside_rectangle_formula(goal, 0, 1, 6))

        at_any_goal = goal_formulas[0]
        for i in range(1,len(goal_formulas)):
            at_any_goal = at_any_goal | goal_formulas[i]

        # Obstacle Avoidance
        obstacle_formulas = []
        for obs in self.obstacles:
            obstacle_formulas.append(outside_rectangle_formula(obs, 0, 1, 6))

        obstacle_avoidance = obstacle_formulas[0]
        for i in range(1, len(obstacle_formulas)):
            obstacle_avoidance = obstacle_avoidance & obstacle_formulas[i]

        # Put all of the constraints together in one specification
        specification = at_any_goal.eventually(0, self.T) & \
                        obstacle_avoidance.always(0, self.T)

        return specification

    def GetSystem(self):
        return DoubleIntegrator(2)

    def add_to_plot(self, ax):
        # Make and add rectangular patches
        for obstacle in self.obstacles:
            ax.add_patch(make_rectangle_patch(*obstacle, color='k', alpha=0.5))
        for goal in self.goals:
            ax.add_patch(make_rectangle_patch(*goal, color='green', alpha=0.5))

        # set the field of view
        ax.set_xlim((0,12))
        ax.set_ylim((0,12))
        ax.set_aspect('equal')
