from .base import BenchmarkScenario
from .common import (inside_rectangle_formula,
                     outside_rectangle_formula,
                     make_rectangle_patch)
from ..systems import DoubleIntegrator

class DoorPuzzle(BenchmarkScenario):
    r"""
    A mobile robot with double integrator dynamics must pick up
    several keys (by visiting certain regions :math:`\mathcal{K}_i`)
    before it cat pass through associated doors (:math:`\mathcal{D}_i`)
    and reach a goal (:math:`\mathcal{G}`). Along the way, it must
    avoid obstacles (:math:`\mathcal{O}_j`).

    .. math::

        \varphi = \bigwedge_{i=1}^{N} \left( \lnot \mathcal{D}_i U_{[0,T]} \mathcal{K}_i \right)  
        \land F_{[0,T]} \mathcal{G} 
        \land G_{[0,T]} (\bigwedge_{j=1}^5 \lnot \mathcal{O}_i)

    :param T:   The time horizon for this scenario
    :param N:   The number of key-door pairs. Must be between 1 and 4.
    """
    def __init__(self, T, N):
        assert N >= 1 and N <= 4, "number of pairs must be between 1 and 4"

        self.N = N
        self.T = T
   
        # Goal location
        self.goal_bounds = (14.1,14.9,4.1,5.9)

        # Obstacle locations
        self.obs1_bounds = (8,15.01,-0.01,4)
        self.obs2_bounds = (8,15.01,6,10.01)
        self.obs3_bounds = (3.5,5,-0.01,2.5)
        self.obs4_bounds = (-0.01,2.5,4,6)
        self.obs5_bounds = (3.5,5,7.5,10.01)

        # Door locations (set to overlap obstacles slightly)
        self.door1_bounds = (12.8,14,3.99,6.01)
        self.door2_bounds = (11.5,12.7,3.99,6.01)
        self.door3_bounds = (10.2,11.4,3.99,6.01)
        self.door4_bounds = (8.9,10.1,3.99,6.01)

        # Key locations
        self.key1_bounds = (1,2,1,2)
        self.key2_bounds = (1,2,8,9)
        self.key3_bounds = (6,7,8,9)
        self.key4_bounds = (6,7,1,2)

    def GetSpecification(self):
        # Goal Reaching
        at_goal = inside_rectangle_formula(self.goal_bounds, 0, 1, 6)

        # Obstacle Avoidance
        not_at_obs1 = outside_rectangle_formula(self.obs1_bounds, 0, 1, 6)
        not_at_obs2 = outside_rectangle_formula(self.obs2_bounds, 0, 1, 6)
        not_at_obs3 = outside_rectangle_formula(self.obs3_bounds, 0, 1, 6)
        not_at_obs4 = outside_rectangle_formula(self.obs4_bounds, 0, 1, 6)
        not_at_obs5 = outside_rectangle_formula(self.obs5_bounds, 0, 1, 6)
        obstacle_avoidance = not_at_obs1 & not_at_obs2 & not_at_obs3 & not_at_obs4 & not_at_obs5

        # Being outside a door region
        no_door1 = outside_rectangle_formula(self.door1_bounds, 0, 1, 6)
        no_door2 = outside_rectangle_formula(self.door2_bounds, 0, 1, 6)
        no_door3 = outside_rectangle_formula(self.door3_bounds, 0, 1, 6)
        no_door4 = outside_rectangle_formula(self.door4_bounds, 0, 1, 6)
       
        # Being in a key region
        key1 = inside_rectangle_formula(self.key1_bounds, 0, 1, 6)
        key2 = inside_rectangle_formula(self.key2_bounds, 0, 1, 6)
        key3 = inside_rectangle_formula(self.key3_bounds, 0, 1, 6)
        key4 = inside_rectangle_formula(self.key4_bounds, 0, 1, 6)
      
        # Not reaching a door until the correspoding key region has been
        # visited
        k1d1 = no_door1.until(key1, 0, self.T)
        k2d2 = no_door2.until(key2, 0, self.T)
        k3d3 = no_door3.until(key3, 0, self.T)
        k4d4 = no_door4.until(key4, 0, self.T)

        # Putting it all together
        if self.N == 1:
            key_constraints = k1d1
        elif self.N == 2:
            key_constraints = k1d1 & k2d2
        elif self.N == 3:
            key_constraints = k1d1 & k2d2 & k3d3
        elif elf.N == 4:
            key_constraints = k1d1 & k2d2 & k3d3 & k4d4

        # Put all of the constraints together in one specification
        specification = obstacle_avoidance.always(0, self.T) & \
                        key_constraints & \
                        at_goal.eventually(0, self.T)

        return specification

    def GetSystem(self):
        return DoubleIntegrator(2)

    def add_to_plot(self, ax):
        # Obstacles
        obs = [self.obs1_bounds, self.obs2_bounds, self.obs3_bounds, self.obs4_bounds, self.obs5_bounds]
        for obs_bounds in obs:
            obs_patch = make_rectangle_patch(*obs_bounds, color='k', alpha=0.5)
            ax.add_patch(obs_patch)
       
        # Doors
        door1 = make_rectangle_patch(*self.door1_bounds, color='red', alpha=0.5)
        door2 = make_rectangle_patch(*self.door2_bounds, color='red', alpha=0.5)
        door3 = make_rectangle_patch(*self.door3_bounds, color='red', alpha=0.5)
        door4 = make_rectangle_patch(*self.door4_bounds, color='red', alpha=0.5)

        # Keys
        key1 = make_rectangle_patch(*self.key1_bounds, color='blue', alpha=0.5)
        key2 = make_rectangle_patch(*self.key2_bounds, color='blue', alpha=0.5)
        key3 = make_rectangle_patch(*self.key3_bounds, color='blue', alpha=0.5)
        key4 = make_rectangle_patch(*self.key4_bounds, color='blue', alpha=0.5)

        # Goals
        goal = make_rectangle_patch(*self.goal_bounds, color='green', alpha=0.5)
        ax.add_patch(goal)

        # Only add the doors and keys that we actually used
        if self.N >= 1:
            ax.add_patch(door1)
            ax.add_patch(key1)

        if self.N >= 2:
            ax.add_patch(door2)
            ax.add_patch(key2)

        if self.N >= 3:
            ax.add_patch(door3)
            ax.add_patch(key3)

        if self.N >= 4:
            ax.add_patch(door4)
            ax.add_patch(key4)

        # set the field of view
        ax.set_xlim((0,15))
        ax.set_ylim((0,10))
        ax.set_aspect('equal')
