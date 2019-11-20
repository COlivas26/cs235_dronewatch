import numpy as np
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

class ScratchLinearRegression:

    def __init__(self, X, y, alpha=0.03, n_iter=1500):

        self.alpha = alpha
        self.n_iter = n_iter
        self.n_samples = len(y)
        self.n_features = np.size(X, 1)
        self.X = np.hstack((np.ones((self.n_samples, 1)), (X - np.mean(X, 0)) / np.std(X, 0)))
        self.y = y[:, np.newaxis]
        self.params = np.zeros((self.n_features + 1, 1))
        self.coef_ = None
        self.intercept_ = None

    def fit(self):

        for i in range(self.n_iter):
            self.params = self.params - (self.alpha / self.n_samples) * self.X.T @ (self.X @ self.params - self.y)

        self.intercept_ = self.params[0]
        self.coef_ = self.params[1:]

        return self

    def score(self, X=None, y=None):

        # if an X or y were not given, then the score will be calculated for the X and y given at initiation
        if X is None:
            X = self.X
        else:
            n_samples = np.size(X,0)
            X = np.hstack((np.ones((n_samples, 1)), (X - np.mean(X, 0)) / np.std(X, 0)))

        if y is None:
            y = self.y
        else:
            y = y[:, np.newaxis]

        y_pred = X @ self.params
        score = 1 - (((y - y_pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())

        return score

    def get_params(self):
        return self.params


# mean squared error used as the cost function
def cost_function(X, y, params):

    num_samples = len(y)

    h = X @ params

    return (1/(2*num_samples)) * np.sum((h-y) ** 2)


def gradient_descent(X, y, params, learning_rate, n_itrs):
    n_samples = len(y)
    J_history = np.zeros((n_itrs, 1))

    for i in range(n_itrs):
        params = params - (learning_rate/n_samples) * X.T @ (X @ params -y)
        J_history[i] = cost_function(X, y, params)

    return J_history, params


# load data
dataset = load_boston()

# region flat Linear Reg implementation
# seperate the data
X = dataset.data
y = dataset.target[:, np.newaxis]

# normalizes the data using standard score
mu = np.mean(X, 0)
sigma = np.std(X, 0)

X = (X-mu) / sigma

# calculcate necessary parameter vales
n_samples = len(y)
X = np.hstack((np.ones((n_samples, 1)), X))
n_features = np.size(X, 1)
params = np.zeros((n_features, 1))

n_iters = 1500
learning_rate = 0.01

# run gradient descent to find optimal w and get final cost
(J_history, optimal_params) = gradient_descent(X, y, params, learning_rate, n_iters)

print("Scratch final cost: ", J_history[-1], "\n")

#endregion

#region Comparison run
# scratch vs scikit-learn implementations
# X = dataset.data
# y = dataset.target
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#
# our_regressor = ScratchLinearRegression(X_train, y_train).fit()
#
# poly_features = PolynomialFeatures(1)
# X_train = poly_features.fit_transform(X_train)
# sklearn_regressor = LinearRegression().fit(X_train, y_train)
#
# our_train_accuracy = our_regressor.score()
# sklearn_train_accuracy = sklearn_regressor.score(X_train, y_train)
#
# our_test_accuracy = our_regressor.score(X_test, y_test)
#
# X_test = poly_features.transform(X_test)
# sklearn_test_accuracy = sklearn_regressor.score(X_test, y_test)
#
# results = pd.DataFrame([[our_train_accuracy, sklearn_train_accuracy],
#              [our_test_accuracy, sklearn_test_accuracy]],
#              ['Training Accuracy', 'Test Accuracy'],
#              ['Our Implementation', 'Sklearn\'s Implementation'])
#
# print(results)
#endregion