import numpy as np
import math

X_bar = np.array([[0, 0, 3, -1, -1, 0, 0, -1], [0, 0, -1, -1, -1, 0, 0, 3]])
X_bar_t = X_bar.transpose()
u_k = np.array([[1/math.sqrt(2), -1/math.sqrt(2)]])

Y = np.dot(u_k, X_bar)

print(np.dot(u_k.transpose(), Y))