import numpy as np


class LinearRegression:

    def __init__(self):

        self.X = []
        self.y = []
        self.theta = []
        self.mu = 0
        self.sigma = 0
        self.theta_log = []

    def set_train_data(self, X, y):

        # normalize and reshape X
        mu = np.mean(X, 0)
        sigma = np.std(X, 0)

        X = (X - mu) / sigma

        self.mu = mu
        self.sigma = sigma
        self.X = np.hstack((np.ones((len(y), 1)), X))
        self.y = y[:, np.newaxis]
        self.theta_log = []

    # MSE or R2 cost function
    def cost_function(self, y_pred, y_actual):
        m = len(y_pred)
        y_pred = y_pred.flatten()
        y_actual = y_actual.flatten()

        return (1/(2 * m))*np.sum((y_pred - y_actual) ** 2)
        # return 1 - (((y_actual - y_pred) ** 2).sum() / ((y_actual - y_actual.mean()) ** 2).sum())

    def gradient_func(self, theta):
        m = len(self.y)
        return (0.03/m) * self.X.T @ (self.X @ theta - self.y)

    # batch gradient descent
    def gradient_descent(self, grad_func, theta, learning_rate, iterations):

        self.theta_log.append(theta)

        for i in range(iterations):
            # theta = theta - (learning_rate * grad_func(theta))
            theta = theta - (grad_func(theta))
            self.theta_log.append(theta)

        self.theta = theta

    def train(self, X, y, learning_rate=0.03, iterations=1500):

        self.set_train_data(X, y)
        theta = np.zeros((X.shape[1] + 1, 1))

        self.gradient_descent(self.gradient_func, theta, learning_rate, iterations)

    def predict(self, X_test):

        X_test = (X_test - self.mu) / self.sigma
        X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
        return X_test @ self.theta

    def cal_costs(self):

        costs = []

        for theta in self.theta_log:

            y_pred = self.X @ theta
            costs.append(self.cost_function(y_pred, self.y))

        return costs
