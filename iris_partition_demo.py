#!/usr/bin/env python

##
#
# An example of partitioning based on the IRIS sampling-based method
#
##

import irispy
import numpy as np
import matplotlib.pyplot as plt

# Problem setup. We'll define the boundaries of several state formulas as lines between adjacent vertices. 
xmin = 0; xmax = 10
ymin = 0; ymax = 10
bounds = irispy.Polyhedron.from_bounds([xmin,ymin],[xmax,ymax])
obstacles = [np.array([[3,3],[3,5]]),
             np.array([[3,5],[8,5]])]

# Choose a starting point to define an initial partition
start = np.array([4,4])

# Find the largest convex region around this starting point
region, debug = irispy.inflate_region(obstacles, start, bounds=bounds, return_debug_data=True)

# Plot the solution
ax = plt.gca()

for obs in obstacles:
    plt.plot(*obs, color='k')
region.getPolyhedron().draw(ax, facecolor='b', edgecolor='b', alpha=0.5)

plt.plot(*start,'go')

plt.xlim((xmin, xmax))
plt.ylim((ymin, ymax))

plt.show()
