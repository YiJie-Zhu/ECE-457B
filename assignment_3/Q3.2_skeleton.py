# Import useful libraries. Feel free to use sklearn.
from sklearn.datasets import fetch_openml
import numpy as np
import heapq
from matplotlib import pyplot as plt


# Load MNIST dataset.
X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
X = np.array(X)

# Conduct PCA to reduce the dimensionality of X.
X_mean = np.mean(X, axis=0)
M = np.tile(X_mean, (len(X), 1))
X_bar = X - M
cov = (1/len(X)) * np.dot(X_bar.transpose(), X_bar) # each row of X_bar is a datapoint, so covariance calculation is reversed

e_vals, e_vecs = np.linalg.eig(cov)

u_k = e_vecs[:, :2]

z_bar = np.dot(u_k.transpose(), X_bar.transpose()).transpose() # transpose z_bar to make it easier to work with

# Visualize the data distribution of digits '0', '1' and '3' in a 2D scatter plot.
digits = ['0', '1', '3']
plt.figure(1)
plt.scatter(z_bar[y == '0', 0], z_bar[y == '0', 1], s=1)
plt.scatter(z_bar[y == '1', 0], z_bar[y == '1', 1], s=1)
plt.scatter(z_bar[y == '3', 0], z_bar[y == '3', 1], s=1)
plt.legend(digits)

# Generate an image of digit '3' using 2D representations of digits '0' and '1'.
mean_0 = np.mean(z_bar[y == '0', :], axis=0)
mean_1 = np.mean(z_bar[y == '1', :], axis=0)

mean_3 = (mean_0 + mean_1) / 2

rep_3 = np.dot(u_k, mean_3.transpose())
image_3 = 1 - rep_3.reshape(28,28) # inversing colour to make shape more clear

plt.figure(2)

plt.imshow(np.real(image_3), cmap='gray')
plt.show()


