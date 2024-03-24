# Import useful libraries. Feel free to use sklearn.
from sklearn.datasets import make_blobs
import math
import random
from collections import defaultdict
import numpy as np
from matplotlib import pyplot as plt


# Construct a 2D toy dataset for clustering.
X, _ = make_blobs(n_samples=1000,
                  centers=[[0, 0], [1, 1], [-1, 1], [-1, -1], [1, -1]],
                  cluster_std=[0.2, 0.3, 0.3, 0.3, 0.3],
                  random_state=26)

# Conduct clustering on X using k-Means, and determine the best k with the elbow method.

def dist(p, c):
    p_x, p_y = p
    c_x, c_y = c
    return math.sqrt(pow(p_x - c_x, 2) + pow(p_y - c_y, 2))

def update_centre(points):
    points_sum = [0, 0]
    for p_x, p_y in points:
        points_sum[0] += p_x
        points_sum[1] += p_y
    
    points_sum[0] = points_sum[0] / len(points)
    points_sum[1] = points_sum[1] / len(points)

    return [points_sum[0], points_sum[1]]

def calc_variance(points_assignment, centres):
    total_points = 0
    total_variance = 0
    for i, points in points_assignment.items():
        total_points += len(points)
        for p in points:
            total_variance += dist(p, centres[i])

    return total_variance / total_points

def compare_centre_diff(new_c, old_c, tol):
    for i in range(len(new_c)):
        if abs(new_c[i] - old_c[i]) > tol:
            return True
        
    return False


# generate centres
k_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
variance_list = []

for k in k_list:
    centres = []
    for _ in range(k):
        centres.append([random.uniform(-1, 1), random.uniform(-1, 1)])
    print(centres)

    points_assignment = defaultdict(list)

    # Continues until the location of centres converge
    while True:

        for p in X:
            dist_from_centres = []
            for c in centres:
                dist_from_centres.append(dist(p, c))

            points_assignment[np.argmin(dist_from_centres)].append(p)

        changed = False
        for i, points in points_assignment.items():
            new_c = update_centre(points)
            changed = changed or compare_centre_diff(new_c, centres[i], 0.001) # Checks if the location of the centres have converged
            centres[i] = new_c

        if not changed:
            break
        points_assignment = defaultdict(list)
    print(centres)
    variance_list.append(1 - calc_variance(points_assignment, centres)) # total variance is the sum of variance for all points and their assigned centroid

plt.plot(k_list, variance_list)
plt.show()

# k=5

k = 5
centres = []
for _ in range(k):
    centres.append([random.uniform(-1, 1), random.uniform(-1, 1)])
print(centres)

points_assignment = defaultdict(list)

# Continues until the location of centres converge
while True:

    for p in X:
        dist_from_centres = []
        for c in centres:
            dist_from_centres.append(dist(p, c))

        points_assignment[np.argmin(dist_from_centres)].append(p)

    changed = False
    for i, points in points_assignment.items():
        new_c = update_centre(points)
        changed = changed or compare_centre_diff(new_c, centres[i], 0.001) # Checks if the location of the centres have converged
        centres[i] = new_c

    if not changed:
        break

    points_assignment = defaultdict(list)

plt.figure(2)
for i, points in points_assignment.items():
    points_np = np.array(points)
    plt.scatter(points_np[:, 0], points_np[:, 1], s=10)

plt.show()
print(points_assignment)


    

