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
  if sorted:
    xind = np.argsort(X[0, :])
    X = X[:, xind]
    Y = Y[:, xind]
  return X, Y

class PolyRegression():
  def __init__(self, degree, epochs = 5000, learning_rate = 0.05):
    self.degree = degree
    self.epochs = epochs
    self.learning_rate = learning_rate

  def get_value(self, X):    
    X_poly = np.hstack([X**i for i in range(self.degree + 1)])
    return X_poly @ self.weights

  def predict(self, X):
    return self.get_value(X)

  def loss(self, x, y):
    return np.mean((y - self.predict(x)) ** 2)

  def fit(self, x, y):
    X_poly = np.hstack([x**i for i in range(self.degree + 1)]) 
    self.weights = np.zeros((self.degree + 1, 1));

    for epoch in range(self.epochs):
      y_pred = X_poly @ self.weights
      error = y_pred - y
      n = x.shape[0]
      gradient = (2 / n) * X_poly.T @ error

      self.weights -= self.learning_rate * gradient

    return self


functions = ['poly1', 'poly2', 'log', 'sin']


XC, YC = gen_data(N=1000, fun='poly2', sigma_noise=0)
plt.plot(XC, YC, '.', label='training data')
for degree in [0, 1, 2, 5]:
    model = PolyRegression(degree).fit(XC, YC)
    plt.plot(XC, model.predict(XC), '-', label=f'poly degree {degree} predicted')
plt.title('Fitting square function without noise and 1000 samples')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.legend()
plt.savefig('fit_poly2.png')
plt.show()

plt.close()

