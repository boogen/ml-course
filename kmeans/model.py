import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

def kmeans(X, k, max_iter=100, tol=1e-4):
  n_samples, n_features = X.shape

  rng = np.random.default_rng()
  initial_indices = rng.choice(n_samples, size=4, replace=False)
  centroids = X[initial_indices]

  for i in range(max_iter):
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis = 2)
    labels = np.argmin(distances, axis=1)

    new_centroids = np.array([X[labels == j].mean(axis = 0) if np.any(labels == j) else centroids[j] for j in range(k)])

    if np.linalg.norm(new_centroids - centroids) < tol:
      break
    
    centroids = new_centroids
  
  return centroids, labels

X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

centroids, labels = kmeans(X, k=4)

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='X')
plt.show()