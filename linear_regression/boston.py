from sklearn.datasets import load_boston
import numpy as np

# Load data
boston = load_boston()
X = boston.data    # shape (506, 13)
y = boston.target  # shape (506,)

# Reshape y to column vector
y = y.reshape(-1, 1)

X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X = (X - X_mean) / X_std

X = np.hstack([np.ones((X.shape[0], 1)), X])

n_samples, n_features = X.shape

weights = np.zeros((n_features, 1))
learning_rate = 0.01
epochs = 1000


def predict(X, weights):
  return X @ weights

def compute_loss(y, y_pred):
  np.mean((y - y_pred) ** 2)


for epoch in range(epochs):
  y_pred = predict(X, weights)
  error = y - y_pred
  gradient = 2 * X.T @ error

  weights -= learning_rate * gradient

  # Print loss occasionally
  if epoch % 100 == 0:
      loss = compute_loss(y, y_pred)
      print(f"Epoch {epoch}, Loss: {loss:.4f}")




  