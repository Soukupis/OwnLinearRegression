import pandas as pd
from sklearn.datasets import load_boston
from LinearRegression import *
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg

dataset = load_boston()
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)

df.head()
print(dataset.target)

np.random.seed(42)

x = dataset.data
y = dataset.target

indices = np.random.permutation(len(x))
test_size = 100

x_train = x[indices[:-test_size]]
y_train = y[indices[:-test_size]]

x_test = x[indices[-test_size:]]
y_test = y[indices[-test_size:]]

from sklearn.linear_model import LinearRegression

regr = LinearRegression()
regr.fit(x_train, y_train)

print("Coeffs: ", regr.coef_)
print("Intercept: ", regr.intercept_)
print("R2: ",regr.score(x_test, y_test))

train_pred =regr.predict(x_train)
test_pred = regr.predict(x_test)

min_val = min(min(train_pred), min(test_pred))
max_val = max(max(train_pred), max(test_pred))


# y_pred = 10, y = 12
# -2
plt.scatter(train_pred, train_pred - y_train, color="blue", s=40)
plt.scatter(test_pred, test_pred - y_test, color="red", s=40)
plt.hlines(y=0, xmin=min_val, xmax=max_val)
plt.show()
