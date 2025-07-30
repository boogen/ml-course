import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

def kmeans(X, k, max_iter=100, tol=1e-4):
  n_samples, n_features = X.shape

  rng = np.random.default_rng()
  centroids = []

  # randomly select the first centroid
  first_idx = np.random.randint(n_samples)
  centroids.append(X[first_idx])

  for _ in range(1, k):
    # compute square distance to the nearest centroid
    dist_sq = np.array([min(np.sum((x - c)**2) for c in centroids) for x in X])
    probs = dist_sq / dist_sq.sum()
    next_idx = np.random.choice(n_samples, p=probs)
    centroids.append(X[next_idx])   
  

  for i in range(max_iter):
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis = 2)
    labels = np.argmin(distances, axis=1)

    new_centroids = np.array([X[labels == j].mean(axis = 0) if np.any(labels == j) else centroids[j] for j in range(k)])

    if np.linalg.norm(new_centroids - centroids) < tol:
      break
    
    centroids = new_centroids
  
  return centroids, labels

X, _ = make_blobs(n_samples=300, centers=4, cluster_std=1.60, random_state=1)

centroids, labels = kmeans(X, k=4)

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='X')
plt.show()