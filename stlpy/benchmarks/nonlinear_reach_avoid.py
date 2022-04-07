from .base import BenchmarkScenario
from .common import inside_circle_formula, make_circle_patch
from ..systems import Unicycle

class NonlinearReachAvoid(BenchmarkScenario):
    r"""
    A 2D mobile robot with unicycle dynamics must
    avoid a circular obstacle (:math:`\mathcal{O}`) before reaching 
    a circular goal (:math:`\mathcal{G}`):

    .. math::

        \varphi = G_{[0,T]} \lnot \mathcal{O} \land F_{[0,T]} \mathcal{G}

    :param goal_center:      a tuple ``(px, py)`` defining the center of the
                             goal region
    :param goal_radius:      a scalar defining the goal radius
    :param obstacle_center:  a tuple ``(px, py)`` defining the center of the
                             obstacle region
    :param obstacle_radius:  a scalar defining the obstacle radius
    :param T:                the time horizon for this scenario.
    """
    def __init__(self, goal_center, goal_radius, obstacle_center, obstacle_radius, T):
        self.goal_center = goal_center
        self.goal_radius = goal_radius

        self.obstacle_center = obstacle_center
        self.obstacle_radius = obstacle_radius

        self.T = T

    def GetSpecification(self):
        # Goal Reaching
        at_goal = inside_circle_formula(self.goal_center, self.goal_radius, 0, 1, 3)

        # Obstacle Avoidance
        at_obstacle = inside_circle_formula(self.obstacle_center,
                self.obstacle_radius, 0, 1, 3)
        not_at_obstacle = at_obstacle.negation()

        # Put all of the constraints together in one specification
        spec = not_at_obstacle.always(0, self.T) & at_goal.eventually(0, self.T)

        return spec

    def GetSystem(self):
        sys = Unicycle(dt=0.1)
        return sys

    def add_to_plot(self, ax):
        # Make and add circular patches
        obstacle = make_circle_patch(self.obstacle_center,
                self.obstacle_radius, color='k', alpha=0.5)
        goal = make_circle_patch(self.goal_center, self.goal_radius, 
                color='green', alpha=0.5)

        ax.add_patch(obstacle)
        ax.add_patch(goal)

        # set the field of view
        ax.set_xlim((0,10))
        ax.set_ylim((0,10))
        ax.set_aspect('equal')
