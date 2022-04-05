import numpy as np

from .base import BenchmarkScenario
from .common import inside_rectangle_formula, make_rectangle_patch
from ..systems import DoubleIntegrator

class SteppingStones(BenchmarkScenario):
    r"""
    A 2D mobile robot with double integrator dynamics must
    navigate to a goal (:math:`\mathcal{G}`) while only stepping 
    on certain predefined spaces (:math:`\mathcal{S}_i`):

    .. math::

        \varphi = G_{[0,T]} \left( \bigvee_{i}^{N_s} \mathcal{S}_i \right) \land
                  F_{[0,T]} \mathcal{G}

    :param num_stones:  The number of stepping stones :math:`N_s`
    :param T:           The specification time horizon
    :param seed:        (optional) The random seed for stone placement.
                        Default is None.
    """
    def __init__(self, num_stones, T, seed=None):
        self.T = T

        # Set the seed for the random number generator (for reproducability)
        np.random.seed(seed)

        # Create the (randomly generated) set of stepping stones
        self.stones = []
        for i in range(num_stones):
            x = np.random.uniform(0,9)  # keep within workspace
            y = np.random.uniform(0,9)
            self.stones.append((x,x+1,y,y+1))

        # Specify the target/goal
        self.target = self.stones[-1]

    def GetSpecification(self):

        # Specify that we must be on any one of the stones
        stone_formulas = []
        for stone in self.stones:
            stone_formulas.append(inside_rectangle_formula(stone, 0, 1, 6))

        on_any_stone = stone_formulas[0]
        for i in range(1, len(stone_formulas)):
            on_any_stone = on_any_stone | stone_formulas[i]

        # Specify that we much reach the target
        reach_target = inside_rectangle_formula(self.target, 0, 1, 6)

        # Put all of the constraints together in one specification
        specification = on_any_stone.always(0, self.T) & \
                        reach_target.eventually(0, self.T)

        return specification

    def GetSystem(self):
        return DoubleIntegrator(2)

    def add_to_plot(self, ax):
        n_stones = len(self.stones)

        # Add rectangles for the stones
        for i in range(n_stones):
            if i < n_stones-1:  # ordinary stepping stones are orange
                ax.add_patch(make_rectangle_patch(*self.stones[i], color='orange', alpha=0.5, zorder=-1))
            else:  # the target is green
                ax.add_patch(make_rectangle_patch(*self.stones[i], color='g', alpha=0.5, zorder=-1))

        # set the field of view
        ax.set_xlim((0,10))
        ax.set_ylim((0,10))
        ax.set_aspect('equal')
