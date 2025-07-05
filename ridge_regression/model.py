import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def gen_data(N=10, sigma_noise=.2, fun='poly2'):
  X = 2 * np.random.rand(N) - 1.0
  X = np.sort(X).reshape(-1, 1)
  eps = np.random.randn(N, 1) * sigma_noise
  if fun=='poly1':
    Y = 0.3 + .7*X + eps
  elif fun=='poly2':
    Y = 0.3 + .7*X - 1.1*X**2 + eps
  elif fun=='log':
    Y = np.log(X+1)**.5 + eps
  elif fun=='sin':
    Y = np.sin(3*np.pi*X) + eps
  else:
    raise ValueError(f"Unknown fun type: {fun}.")
  return X, Y

class PolyRegression():
  def __init__(self, degree, epochs = 5000, learning_rate = 0.05, reg_lambda=0.0):
    self.degree = degree
    self.epochs = epochs
    self.learning_rate = learning_rate
    self.reg_lambda = reg_lambda

  def get_value(self, X):    
    X_poly = np.hstack([X**i for i in range(self.degree + 1)])
    return X_poly @ self.weights

  def predict(self, X):
    return self.get_value(X)

  def loss(self, x, y):
    reg_term = self.reg_lambda * np.sum(self.weights[1:] ** 2) 
    return np.mean((y - self.predict(x)) ** 2) + reg_term

  def fit(self, x, y):
    X_poly = np.hstack([x**i for i in range(self.degree + 1)]) 
    self.weights = np.zeros((self.degree + 1, 1));

    for epoch in range(self.epochs):
      y_pred = X_poly @ self.weights
      error = y_pred - y
      reg_term = self.reg_lambda * self.weights
      n = x.shape[0]
      reg_term = self.reg_lambda * self.weights
      reg_term[0] = 0  # don't regularize bias
      gradient = (2 / n) * (X_poly.T @ error + reg_term)
      self.weights -= self.learning_rate * gradient

    return self

  def solve(self, x, y):
    # Build polynomial feature matrix
    X_poly = np.hstack([x**i for i in range(self.degree + 1)])

    XT_X = X_poly.T @ X_poly
    XT_y = X_poly.T @ y
    I = np.eye(X_poly.shape[1])
    I[0, 0] = 0  # don't regularize bias
    self.weights = np.linalg.inv(XT_X + self.reg_lambda * I) @ XT_y

    return self


functions = ['poly1', 'poly2', 'log', 'sin']


X_train, Y_train = gen_data(N=10, fun='poly2', sigma_noise=0.2)
X_test, Y_test = gen_data(N=1000, fun='poly2', sigma_noise=0.2)
plt.plot(X_train, Y_train, '.', label='training data')
for degree in [2, 10, 20]:
    model = PolyRegression(degree, epochs=5000, learning_rate=0.01, reg_lambda=0.05).solve(X_train, Y_train)
    plt.plot(X_test, model.predict(X_test), '-', label=f'poly degree {degree} predicted')
plt.title('Small dataset with noise and model with regulation')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.legend()
plt.show()

plt.close()

