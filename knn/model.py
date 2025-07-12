import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

class KNNClassifier:
  def __init__(self, k=3):
    self.k = k

  def fit(self, X, y):
    self.X_train = X
    self.y_train = y

  def _euclidean_distance(self, x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

  def predict(self, X):
    predictions = [self._predict_single(x) for x in X]
    return np.array(predictions)

  def _predict_single(self, x):
    distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]

    k_indices = np.argsort(distances)[:self.k]

    k_neighbor_labels = [self.y_train[i] for i in k_indices]

    most_common = Counter(k_neighbor_labels).most_common(1)
    return most_common[0][0]

  def score(self, X, y):
    y_pred = self.predict(X)
    return np.mean(y_pred == y)

  
iris = pd.read_csv('iris.csv')

X = iris.drop('target', axis=1).values
y = iris['target'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

knn = KNNClassifier(k=5)
knn.fit(X_train, y_train)
accuracy = knn.score(X_test, y_test)

print(f"KNN Classifier Accuracy: {accuracy:.2f}")

sklearn_knn = KNeighborsClassifier(n_neighbors=5)
sklearn_knn.fit(X_train, y_train)
print("SKLearn KNN Accuracy:", sklearn_knn.score(X_test, y_test))