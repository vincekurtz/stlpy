#!/usr/bin/env python

##
#
# An example of partitioning based on the IRIS sampling-based method
#
##

import irispy
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(3)

def plot_solution(region, obstacles, start):
    """
    Convienience fuction for making a nice plot of
    the scenario (obstacles), start point, and convex partition (region).
    """
    ax = plt.gca()

    for obs in obstacles:
        if obs.shape[1] <= 2:
            plt.plot(*obs, color='k')
        else:
            irispy.drawing.draw_convhull(obs.T, ax, edgecolor='k', facecolor='k', alpha=0.5)

    region.getPolyhedron().draw(ax, facecolor='b', edgecolor='b', alpha=0.5)

    plt.plot(*start,'go')

    plt.xlim((xmin, xmax))
    plt.ylim((ymin, ymax))

    plt.show()

# Problem setup. We'll define the boundaries of several state formulas as lines between adjacent vertices. 
xmin = 0; xmax = 10
ymin = 0; ymax = 10
bounds = irispy.Polyhedron.from_bounds([xmin,ymin],[xmax,ymax])
obstacles = [np.array([[3,3],[4,6]]),
             np.array([[3,5],[4,4]]),
             np.array([[3,5],[6,6]]),
             np.array([[5,5],[4,6]]),
             np.array([[4,4],[5,9]]),
             np.array([[8,8],[5,9]]),
             np.array([[4,8],[5,5]]),
             np.array([[4,8],[9,9]])]

prob = irispy.IRISProblem(2)  # 2d
prob.setBounds(bounds)
for obs in obstacles:
    prob.addObstacle(obs)
options = irispy.IRISOptions()
options.require_containment = True

existing_partitions = []

for i in range(13):
    print(i)
    # Choose a starting point to define an initial partition
    start = np.random.uniform(low=0, high=10, size=2)

    # Make sure this starting point is in free spacy
    while any([p.contains(start,0) for p in existing_partitions]):
        start = np.random.uniform(low=0, high=10, size=2)

    # Find the largest convex region around this starting point
    prob.setSeedPoint(start)
    region = irispy.iris_wrapper.inflate_region(prob, options)
    #region = irispy.inflate_region(obstacles, start, bounds=bounds, require_containment=True)
    poly = region.getPolyhedron()

    # Show this convex region
    plot_solution(region, obstacles, start)

    # Add this region to the list of obstacles
    obstacles.append(np.asarray(poly.generatorPoints()).T)
    prob.addObstacle(np.asarray(poly.generatorPoints()).T)
    existing_partitions.append(poly)

plot_solution(region, obstacles, start)
