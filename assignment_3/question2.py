import math

points = [
    (0, 1),
    (3, 3),
    (1, 1),
    (2, 3),
    (1, 0),
    (0, 0),
    (3, 2),
    (2, 2)
]

c = [
    (1/2, 1/2),
    (5/2, 5/2)
]

def dist(c, p):
    c_x, c_y = c
    p_x, p_y = p

    return math.sqrt(pow(c_x - p_x, 2) + pow(c_y - p_y, 2))

for centroid in c:
    print(f"For centroid {centroid}: ")
    for p in points:
        print(f"Point {p}: {dist(centroid, p)}")