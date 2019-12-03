import numpy as np
from sklearn.preprocessing import PolynomialFeatures


def poly_row_covert(row, degree):

    return [xi ** degree for xi in row]


def convert_poly(arr, degree):

    poly_dict = dict()

    for i in range(1, degree + 1):

        poly = np.apply_along_axis(poly_row_covert, 1, X, i)

        poly_dict[i] = poly

    return poly_dict


def poly_feat(arr):

    poly_arr = []

    for row in arr:

        tmp_row = []
        for i in range(len(row)):

            tmp_row.append(row[i])
            for j in range((len(row) - i)):
                tmp_row.append(row[i] * row[i + j])

        poly_arr.append(tmp_row)

    poly_arr = np.array(poly_arr)
    return np.hstack((np.ones((poly_arr.shape[0], 1)), poly_arr))


#
# poly = PolynomialFeatures(2)
# sk_poly = poly.fit_transform(X)
# # poly_X = convert_poly(X)
#

X = np.array([[1, 2, 3], [4, 5, 6]])

poly_X = poly_feat(X)
#
# for i in range(len(X)):
#
#     new_X.append(X[i])
#     for j in range((len(X) - i)):
#         new_X.append(X[i] * X[i+j])


print()

