import numpy as np


class LogisticRegression:

    def __init__(self):

        self.X = []
        self.y = []
        self.theta = []
        self.mu = 0
        self.sigma = 0

    def set_train_data(self, X, y):

        # normalize and reshape X
        mu = np.mean(X, 0)
        sigma = np.std(X, 0)

        X = (X - mu) / sigma

        self.mu = mu
        self.sigma = sigma
        self.X = np.hstack((np.ones((len(y), 1)), X))
        self.y = y[:, np.newaxis]

    # sigmoid function that takes in theta(weights) and x
    def sigmoid(self, theta, X):
        return 1 / (1 + np.exp(- (X @ theta)))

    # cost function taken from the sigmoid function
    def cost_function(self, theta):
        m = self.X.shape[0]
        cost = -(1 / m) * np.sum(self.y * np.log(self.sigmoid(theta))
                                 + (1 - self.y) * np.log(1 - self.sigmoid(theta)))
        return cost

    # gradient function, derivative of the cost function and then simplified
    def gradient_func(self, theta):
        m = self.X.shape[0]
        return (1 / m) * np.dot(self.X.T, self.sigmoid(theta, self.X) - self.y)

    # batch gradient descent
    def gradient_descent(self, grad_func, theta, learning_rate=0.01, n_itrs=1500):
        for i in range(n_itrs):
            theta = theta - (learning_rate * grad_func(theta))

        self.theta = theta

    def train(self, X, y):

        self.set_train_data(X, y)
        theta = np.zeros((X.shape[1] + 1, 1))

        self.gradient_descent(self.gradient_func, theta)

    # predict the classes for the data
    def predict(self, X_test):

        X_test = (X_test - self.mu) / self.sigma
        X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
        return np.round(self.sigmoid(self.theta, X_test))

    # calculate the percentage of accuracy of prediction
    def accuracy(self, y_pred, y_actual):
        y_pred = y_pred.flatten()
        y_actual = y_actual.flatten()

        accu = np.mean(y_pred == y_actual)
        return accu * 100


