from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
import numpy as np

# Load data
housing = fetch_california_housing()

X = housing.data
y = housing.target

# Reshape y to column vector
y = y.reshape(-1, 1)

# Normalize features
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X = (X - X_mean) / X_std

# Add bias term (column of ones)
X = np.hstack([np.ones((X.shape[0], 1)), X])

n_samples, n_features = X.shape

# Custom Gradient Descent Implementation
weights = np.zeros((n_features, 1))
learning_rate = 0.01
epochs = 1001


def predict(X, weights):
  return X @ weights

def compute_loss(y, y_pred):
  return np.mean((y - y_pred) ** 2)


# Train the model using gradient descent
for epoch in range(epochs):
  y_pred = predict(X, weights)
  error = y_pred - y
  gradient = 2 * X.T @ error / n_samples

  weights -= learning_rate * gradient

  # Print loss occasionally
  if epoch % 100 == 0:
      loss = compute_loss(y, y_pred)
      print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Train the model using sklearn's LinearRegression
sklearn_model = LinearRegression()
sklearn_model.fit(X[:, 1:], y)  # Skip the bias term for sklearn

# Predict using sklearn's model
sklearn_predictions = sklearn_model.predict(X[:, 1:])

# Compute loss for sklearn model
sklearn_loss = np.mean((y - sklearn_predictions) ** 2)
print(f"Sklearn Model - Loss: {sklearn_loss:.4f}")

  