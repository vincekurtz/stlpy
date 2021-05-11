#!/usr/bin/env python

##
#
# An example of partitioning based on the IRIS sampling-based method
#
##

import cdd
import irispy
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(3)

def plot_solution(poly, obstacles, start):
    """
    Convienience fuction for making a nice plot of
    the scenario (obstacles), start point, and convex partition (poly).
    """
    ax = plt.gca()

    for obs in obstacles:
        if obs.shape[1] <= 2:
            plt.plot(*obs, color='k')
        else:
            irispy.drawing.draw_convhull(obs.T, ax, edgecolor='k', facecolor='k', alpha=0.5)

    poly.draw(ax, facecolor='b', edgecolor='b', alpha=0.5)

    plt.plot(*start,'go')

    plt.xlim((xmin-0.1, xmax+0.1))
    plt.ylim((ymin-0.1, ymax+0.1))

    plt.show()

# Problem setup. We'll define the boundaries of several state formulas as lines between adjacent vertices. 
xmin = 0; xmax = 10
ymin = 0; ymax = 10
bounds = irispy.Polyhedron.from_bounds([xmin,ymin],[xmax,ymax])
#obstacles = [np.array([[3,3],[4,6]]),
#             np.array([[3,5],[4,4]]),
#             np.array([[3,5],[6,6]]),
#             np.array([[5,5],[4,6]]),
#             np.array([[4,4],[5,9]]),
#             np.array([[8,8],[5,9]]),
#             np.array([[4,8],[5,5]]),
#             np.array([[4,8],[9,9]])]
state_formulas = [np.array([[3,3,5,5],[4,6,4,6]]),
                  np.array([[4,4,8,8],[5,9,5,9]])]

partitions = []

def generate_partition(state_formulas, existing_partitions, bounds):
    """
    Given lists of existing state formulas and existing partitions,
    generate and return a new partition. 

    Existing partitions are treated as obstacles, and starting points
    are selected outside of these. State formulas are not, however, since
    we sometimes want to generate a partition inside a state formula.

    Returns the new partition and the starting point.
    """
    # Choose a starting point to define an initial partition
    start = np.random.uniform(low=0, high=10, size=2)

    # If the sampled point is inside an existing partition, we'll ignore it
    # and use a new sample
    if any([p.contains(start,0) for p in existing_partitions]):
        return generate_partition(state_formulas, existing_partitions, bounds)

    # If the sampled point is inside one or more state formulas, we'll consider
    # the intersection of those state formulas as the underlying bounds
    obstacles = []
    for formula in state_formulas:
        # Use cdd to convert from vertex description to halfspace description
        V = formula.T
        t = np.ones((formula.shape[1],1))
        tV = np.hstack([t,V])
        mat = cdd.Matrix(tV)
        mat.rep_type = cdd.RepType.GENERATOR
        poly = cdd.Polyhedron(mat).get_inequalities()
        bA = np.asarray(poly)
        b = bA[:,0]
        A = -bA[:,1:]

        # Create a corresponding irispy Polyhedron and update the bounds
        # if necessary
        p = irispy.Polyhedron(A, b)
        if p.contains(start,0):
            bounds.appendConstraints(p)
        else:
            # If the starting point isn't in this formula, we'll treat this formula
            # as an obstacle
            obstacles.append(formula)

    # Add existing partitions as obstacles
    for p in existing_partitions:
        obstacles.append(np.asarray(p.generatorPoints()).T)

    # Run IRIS to find a large region of free space
    region = irispy.inflate_region(obstacles, start, bounds=bounds, require_containment=True)

    return region.getPolyhedron(), start


for i in range(50):
    bounds = irispy.Polyhedron.from_bounds([xmin,ymin],[xmax,ymax])
    new_partition, start = generate_partition(state_formulas, partitions, bounds)

    obstacles = state_formulas
    plot_solution(new_partition, obstacles, start)
    
    partitions.append(new_partition)
