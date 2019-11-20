
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import fmin_tnc


def load_data(path, header):
    marks_df = pd.read_csv(path, header=header)
    return marks_df

def sigmoid(x):
    # Activation function used to map any real value between 0 and 1
    return 1 / (1 + np.exp(-x))


def net_input(theta, x):
    # Computes the weighted sum of inputs
    return np.dot(x, theta)


def probability(theta, x):
    # Returns the probability after passing through sigmoid
    return sigmoid(net_input(theta, x))


def lrloss_reg(w, X, Y, lmbda):
    ## !!Your code here!!
    sum1 = 0
    for i in range(X.shape[0]):
        yi = Y[i]
        xi = np.array([X[i]])
        sum1 += np.log((1 + np.exp(-yi * np.matmul(xi, w))))

    sum2 = 0
    for j in range(len(w))[1:]:
        sum2 += w[j] ** 2
    sum2 *= (lmbda / 2)

    return sum1 + sum2


def lrloss(w, X, Y):
    sum1 = 0
    for i in range(X.shape[0]):
        yi = Y[i]
        xi = np.array([X[i]])
        sum1 += np.log((1 + np.exp(np.matmul(xi, w))))

    return -sum1 / X.shape[0]


def loss(theta, x, y):
    m = x.shape[0]
    total_cost = -(1 / m) * np.sum(
        y * np.log(probability(theta, x)) + (1 - y) * np.log(
            1 - probability(theta, x)))
    return total_cost

def lrgrad_reg(w, X, Y, lmbda):
    ## !!Your code here!!
    sum1 = 0
    for i in range(X.shape[0]):
        yi = Y[i]
        xi = np.array([X[i]])
        sum1 += (yi * xi) / (np.exp(yi * np.matmul(xi, w)) + 1)
    sum1 *= -1

    sum2 = 0
    for j in range(len(w))[1:]:
        sum2 += w[j]
    sum2 *= lmbda

    return sum1 + sum2


def lrgrad(w, X, Y):
    sum1 = 0
    for i in range(X.shape[0]):
        yi = Y[i]
        xi = np.array([X[i]])
        sum1 += (xi) / (np.exp(np.matmul(xi, w)) + 1)
    sum1 *= -1

    return sum1 / X.shape[0]


def gradient(theta, x, y):
    # Computes the gradient of the cost function at the point theta
    m = x.shape[0]
    return (1 / m) * np.dot(x.T, sigmoid(net_input(theta,   x)) - y)


def lrhess(w, X, Y, lmbda):
    ## !!Your code here!!
    sum1 = 0
    for i in range(X.shape[0]):
        yi = Y[i]
        xi = np.array([X[i]])
        sum1 += (np.exp(-yi * np.matmul(xi, w)) / ((1 + np.exp(-yi * np.matmul(xi, w))) ** 2)) * (
            np.matmul(xi.transpose(), xi))
    sum1 += (lmbda * (len(w) - 1))

    return sum1


def newton(w, fn, gradfn, hessfn, ittfn=None):
    newton_flag = True
    grad_flag = True
    eta = 1

    counter = 0

    while newton_flag:

        counter += 1

        fn_result = fn(w)
        grad_result = gradfn(w)
        hess_result = hessfn(w)
        hess_result_inv = np.linalg.inv(hess_result)

        newton_result = np.matmul(hess_result_inv, np.transpose(grad_result))
        new_w = w - newton_result
        new_fn_result = fn(new_w)

        if counter == 8:
            print()

        if np.isinf(new_fn_result).any() or (new_fn_result > fn_result or new_fn_result == fn_result):

            eta *= 2
            fn_result_sub = fn_result

            while True:
                # fn_result = new_fn_result
                # w = new_w
                grad_result = gradfn(w)
                new_w = np.subtract(w, (eta * grad_result))
                new_fn_result = fn(new_w)

                if fn_result_sub.shape[1] == 1 or (np.isinf(fn_result_sub).any() and not np.isinf(new_fn_result).any()):
                    fn_result_sub = new_fn_result
                else:
                    if np.greater(new_fn_result, fn_result_sub).any() or np.equal(new_fn_result, fn_result_sub).all()\
                            or np.isinf(new_fn_result).any():
                        if eta < 10 ** -10:
                            newton_flag = False
                            break
                        else:
                            eta /= 2
                    else:
                        break


        if newton_flag:
            w = new_w

        if w.shape[1] > 1:
            print()
    return w


def fit(x, y, theta):
    opt_weights = fmin_tnc(func=lrloss, x0=theta, fprime=lrgrad, args=(x, y.flatten()))
    return opt_weights[0]


# lmbdas = [10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1, 1, 10, 100]
#lmbdas = [10**-1, 1, 10, 100]
#
# errors = []
#
# x_len = len(trainX)
# part_size = int(x_len / 3)
# fold_errors = []
#
# x_train = []
# y_train = []
# x_test = []
# y_test = []
#
#
# y_result = []
# for x in x_test:
#     p = 1 / (1 + np.exp(-1 * np.matmul(np.transpose(x), new_w)))
#     y_result.append(1 if p >= 0.5 else 0)
#
# y_len = len(y_test)
#
# correct = 0
# for i in range(y_len):
#
#     if y_result[i] == y_test[i]:
#         correct += 1
#
# error_rate = ((y_len - correct) / y_len) * 100
# fold_errors.append(error_rate)
#
# err_sum = 0
# for err in fold_errors:
#     err_sum += err
#
# errors.append(err_sum / 3)


# load the data from the file
data = load_data("data.txt", None)

# X = feature values, all the columns except the last column
X = data.iloc[:, :-1]

# y = target values, last column of the data frame
y = data.iloc[:, -1]

# filter out the applicants that got admitted
admitted = data.loc[y == 1]

# filter out the applicants that din't get admission
not_admitted = data.loc[y == 0]

# plots
plt.scatter(admitted.iloc[:, 0], admitted.iloc[:, 1], s=10, label='Admitted')
plt.scatter(not_admitted.iloc[:, 0], not_admitted.iloc[:, 1], s=10, label='Not Admitted')

X = np.c_[np.ones((X.shape[0], 1)), X]
y = y[:, np.newaxis]
theta = np.zeros((X.shape[1], 1))

parameters = fit(X, y, theta)

x_values = [np.min(X[:, 1] - 5), np.max(X[:, 2] + 5)]
y_values = -(parameters[0] + np.dot(parameters[1], x_values)) / parameters[2]

plt.plot(x_values, y_values, label='Decision Boundary')
plt.xlabel('Marks in 1st Exam')
plt.ylabel('Marks in 2nd Exam')
plt.legend()
plt.show()


