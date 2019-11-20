
# imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import fmin_tnc
from sklearn.datasets import make_classification

def load_data(path, header):
    marks_df = pd.read_csv(path, header=header)
    return marks_df


# sigmoid function that takes in theta(weights) and x
def sigmoid(theta, x):
    return 1 / (1 + np.exp(- np.dot(x, theta)))


# cost function taken from the sigmoid function
def cost_function(theta, x, y):
    m = x.shape[0]
    cost = -(1 / m) * np.sum(y * np.log(sigmoid(theta, x)) + (1 - y) * np.log(1 - sigmoid(theta, x)))
    return cost


# gradient function, derivative of the cost function and then simplified
def gradient(theta, x, y):
    m = x.shape[0]
    return (1 / m) * np.dot(x.T, sigmoid(theta,   x) - y)


def fit(x, y, theta):
    opt_weights = fmin_tnc(func=cost_function, x0=theta, fprime=gradient, args=(x, y.flatten()))
    return opt_weights[0]


def predict(x):
    theta = parameters[:, np.newaxis]
    return sigmoid(theta, x)


def accuracy(x, actual_classes, probab_threshold=0.5):
    predicted_classes = (predict(x) >=
                         probab_threshold).astype(int)
    predicted_classes = predicted_classes.flatten()
    accu = np.mean(predicted_classes == actual_classes)
    return accu * 100


def gradient_descent(grad_func, X, y, params, learning_rate=0.01, n_itrs=1500):

    for i in range(n_itrs):
        params = params - (learning_rate * grad_func(params, X, y))

    return params


if __name__ == "__main__":
    # load the data from the file
    # data = load_data("data.txt", None)
    #
    # # X = feature values, all the columns except the last column
    # X = data.iloc[:, :-1]
    #
    # # y = target values, last column of the data frame
    # y = data.iloc[:, -1]
    #
    X, y = make_classification(n_samples=500, n_features=2, n_redundant=0, n_informative=1,
                               n_clusters_per_class=1, random_state=14)
    #
    #
    # filter out the applicants that got admitted
    # admitted = data.loc[y == 1]
    #
    # # filter out the applicants that din't get admission
    # not_admitted = data.loc[y == 0]

    #plots
    # plt.scatter(admitted.iloc[:, 0], admitted.iloc[:, 1], s=10, label='Admitted')
    # plt.scatter(not_admitted.iloc[:, 0], not_admitted.iloc[:, 1], s=10, label='Not Admitted')
    # plt.legend()
    # plt.show()
    #
    plt.scatter(X[:, 0], X[:, 1], label='Points')
    # plt.show()
    #

    # normalizes the X
    mu = np.mean(X, 0)
    sigma = np.std(X, 0)
    X = (X-mu) / sigma

    X = np.hstack((np.ones((len(y), 1)), X))

    # X = np.c_[np.ones((X.shape[0], 1)), X]
    y = y[:, np.newaxis]
    theta = np.zeros((X.shape[1], 1))
    #
    # parameters = fit(X, y, theta)
    parameters = gradient_descent(gradient, X, y, theta, n_itrs=2000)
    #
    # x_values = [np.min(X[:, 1] - 5), np.max(X[:, 2] + 5)]
    # y_values = - (parameters[0] + np.dot(parameters[1], x_values)) / parameters[2]
    # #
    # plt.plot(x_values, y_values, label='Decision Boundary')
    # plt.xlim((-3, 3))
    # plt.ylim((-3, 3))
    # plt.legend()
    # plt.show()

    y_pred = np.round(sigmoid(parameters, X))
    # y_pred = y_pred[:, np.newaxis]
    score = float(sum(y_pred == y)) / float(len(y))

    print(score)
