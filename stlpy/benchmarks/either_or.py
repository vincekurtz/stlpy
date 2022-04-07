from .base import BenchmarkScenario
from .common import (inside_rectangle_formula,
                     outside_rectangle_formula,
                     make_rectangle_patch)
from ..systems import DoubleIntegrator

class EitherOr(BenchmarkScenario):
    r"""
    A 2D mobile robot with double integrator dynamics must
    avoid an obstacle (:math:`\mathcal{O}`) before reaching a goal
    (:math:`\mathcal{G}`). Along the way, the robot must reach one
    of two intermediate targets (:math:`\mathcal{T}_i`) and stay
    there for several timesteps:

    .. math::

        \varphi = 
            F_{[0,T-\tau]} 
                \left( G_{[0,\tau]} \mathcal{T}_1 \lor G_{[0,\tau]} \mathcal{T}_2 \right)
            \land F_{[0,T]} \mathcal{G} 
            \land G_{[0,T]} \lnot \mathcal{O}

    :param goal:        Tuple containing bounds of the rectangular goal region
    :param target_one:  Tuple containing bounds of the rectangular first target
    :param target_two:  Tuple containing bounds of the rectangular second target
    :param obstacle:    Tuple containing bounds of the rectangular obstacle
    :param T:           Total number of time steps
    :param T_dwell:     Dwell time :math:`\tau` (integer number of timesteps)
    """
    def __init__(self, goal, target_one, target_two, obstacle, T, T_dwell):
        self.goal = goal
        self.target_one = target_one
        self.target_two = target_two
        self.obstacle = obstacle
        self.T = T
        self.T_dwell = T_dwell

    def GetSpecification(self):
        # Goal Reaching
        at_goal = inside_rectangle_formula(self.goal, 0, 1, 6)

        # Target reaching
        at_target_one = inside_rectangle_formula(self.target_one, 0, 1, 6).always(0, self.T_dwell)
        at_target_two = inside_rectangle_formula(self.target_two, 0, 1, 6).always(0, self.T_dwell)
        at_either_target = at_target_one | at_target_two

        # Obstacle Avoidance
        not_at_obstacle = outside_rectangle_formula(self.obstacle, 0, 1, 6)

        specification = at_either_target.eventually(0, self.T-self.T_dwell) & \
                        not_at_obstacle.always(0, self.T) & \
                        at_goal.eventually(0, self.T)

        return specification

    def GetSystem(self):
        return DoubleIntegrator(2)

    def add_to_plot(self, ax):
        # Make and add rectangular patches
        ax.add_patch(make_rectangle_patch(*self.obstacle, color='k', alpha=0.5))
        ax.add_patch(make_rectangle_patch(*self.target_one, color='blue', alpha=0.5))
        ax.add_patch(make_rectangle_patch(*self.target_two, color='blue', alpha=0.5))
        ax.add_patch(make_rectangle_patch(*self.goal, color='green', alpha=0.5))

        # set the field of view
        ax.set_xlim((0,10))
        ax.set_ylim((0,10))
        ax.set_aspect('equal')
