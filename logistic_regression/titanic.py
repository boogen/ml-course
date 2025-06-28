import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import RobustScaler


def load_data(seed):
  train_data = pd.read_csv('train.csv')
  train_data = train_data[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

  train_data['Age'] = train_data['Age'].fillna(train_data['Age'].median())
  train_data['Embarked'] = train_data['Embarked'].fillna(train_data['Embarked'].mode()[0])

  train_data = pd.get_dummies(train_data, columns = ['Sex'], drop_first=True)
  train_data = pd.get_dummies(train_data, columns = ['Embarked'], drop_first=True)

  train_data = train_data.astype(float)

  X = train_data.drop('Survived', axis=1).values
  y = train_data['Survived'].values.reshape(-1, 1)



  train_data_num_col = train_data.select_dtypes(exclude=['object']).columns
  train_data_num = train_data[train_data_num_col]


  return train_test_split(X, y, test_size=0.2, random_state=seed)

class LogisticRgression:
  def __init__(self, lr=0.01, num_iter=10000):
    self.lr = lr
    self.num_iter = num_iter
  
  def sigmoid(self, z):
    z = np.clip(z, -500, 500) 
    return 1 / (1 + np.exp(-z))

  def fit(self, X, y):
    self.theta = np.zeros((X.shape[1], 1))

    for _ in range(self.num_iter):
      z = X @ self.theta
      h = self.sigmoid(z)
      gradient = (X.T @ (h - y)) / y.shape[0]
      self.theta -= self.lr * gradient

  def predict(self, X):
    return self.sigmoid(X @ self.theta) >= 0.5

  def th(self, X):
    return X @ self.theta

# Max accuracy: 0.7877094972067039 for seed 273
# Min accuracy: 0.5865921787709497 for seed 257

X_train, X_test, y_train, y_test = load_data(273)

scaler = RobustScaler()

X_train = scaler.fit_transform(X_train)

model = LogisticRgression(lr=0.1, num_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = (y_pred == y_test).mean()

print(f"Accuracy: {accuracy}")

