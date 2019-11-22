import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models.lin_reg import LinearRegression as ScratchLinearRegression
from models.log_reg import LogisticRegression as ScratchLogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.datasets import load_boston

from sklearn.preprocessing import StandardScaler

# example of the logistic regression model
# X, y = make_classification(n_samples=3000, n_features=2, n_redundant=0, n_informative=1,
#                            n_clusters_per_class=1, random_state=14)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# logistic_reg = ScratchLogisticRegression()
# logistic_reg.train(X_train, y_train, iterations=2000)
# y_pred = logistic_reg.predict(X_test)
# accuracy = logistic_reg.accuracy(y_pred, y_test)
# print('Prediction accuracy: %f' % accuracy)
#
# # show plot of convergence of costs when calculating theta
# costs = logistic_reg.cal_costs()
# plt.plot(range(len(costs)), costs)
# plt.title('Convergence of cost function')
# plt.xlabel('iteration')
# plt.ylabel('cost')
# plt.show()

# example of the linear regression model
X, y = load_boston(return_X_y=True)
dataset = load_boston()
X = dataset.data
y = dataset.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

linear_reg = ScratchLinearRegression()
linear_reg.train(X_train, y_train)
y_pred = linear_reg.predict(X_test)
score = linear_reg.cost_function(y_pred, y_test)
print('MSE Score: %f' % score)

# show plot of convergence of costs when calculating theta
costs = linear_reg.cal_costs()
plt.plot(range(len(costs)), costs)
plt.title('Convergence of cost function')
plt.xlabel('iteration')
plt.ylabel('cost')
plt.show()