import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from imblearn.over_sampling import SMOTE
from models.lin_reg import LinearRegression as ScratchLinearRegression
from models.log_reg import LogisticRegression as ScratchLogisticRegression
from sklearn.metrics import confusion_matrix, f1_score

from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split


# read in the data

# lin_reg_data = pd.read_csv("linRegressionData.csv")
# # limit to flights that are greater than 1 min and less than 6.5 hours
# lin_reg_data = lin_reg_data.loc[lin_reg_data.apply(lambda x: (60 < x['flight_duration'] < 23400), axis=1)]
# # # lin_reg_data = lin_reg_data.loc[lin_reg_data.apply(lambda x: x['rfid'] == '00 A6 92 05 08 00 12 E0', axis=1)]
# #
# lin_X = lin_reg_data.iloc[:, 1:-1]
# lin_y = lin_reg_data.iloc[:, -1]
# lin_X = lin_X.drop(columns=['nosema'])
# # separate the datasets into train and test sets
# X_train_lin, X_test_lin, y_train_lin, y_test_lin = train_test_split(lin_X, lin_y, test_size=0.3, random_state=42)

# # apply the Linear Regression Model
# # 
# # analyze the data, print out a correlation map of features to the
# # lin_reg_data_corr = (lin_reg_data.iloc[:, 1:]).corr()
# # sns.heatmap(lin_reg_data_corr,
# #             annot=True,
# #             cmap="RdGy",
# #             linewidths=0.3,
# #             annot_kws={"size": 8})
# # plt.xticks(rotation=90)
# # plt.yticks(rotation=0)
# # plt.show()
# 
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(lin_X['humidity'], lin_X['wind_speed'], (lin_y / 60))
# ax.set_xlabel('humidity (percent')
# ax.set_ylabel('wind speed (mph)')
# ax.set_zlabel('flight duration (minutes)')
# plt.scatter(lin_X['humidity'], (lin_y/60))
# plt.ylabel('flight duration (min)')
# plt.xlabel('humidity (percent)')
# plt.title('Flight duration vs. Humidity')
# plt.show()

print()
# linear_reg = ScratchLinearRegression()
# linear_reg.train(X_train_lin, y_train_lin, iterations=200)
# y_pred_lin = linear_reg.predict(X_test_lin)
# rmse_score = (linear_reg.cost_function(y_pred_lin, y_test_lin)) ** 0.5
# print('RMSE Score: %f' % rmse_score)
#
# # show plot of convergence of costs when calculating theta
# lin_costs = linear_reg.cal_costs()
# plt.plot(range(len(lin_costs)), lin_costs)
# plt.title('Convergence of cost function')
# plt.xlabel('iteration')
# plt.ylabel('cost')
# plt.show()
#
# print()
#
log_reg_data = pd.read_csv("logRegressionData.csv")
log_reg_data = log_reg_data.loc[log_reg_data.apply(lambda x: 180 < x['flight_duration'] < 2880, axis=1)]
log_X = log_reg_data.iloc[:, 1:-1]
log_y = log_reg_data.iloc[:, -1]
log_X = log_X.drop(columns=['nosema'])

# split the data into train and test sets
X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(log_X, log_y, test_size=0.2, random_state=42)

# split the training data into train and validation sets
X_train_log, X_val_log, y_train_log, y_val_log = train_test_split(X_train_log, y_train_log, test_size=0.25, random_state=42)

sm = SMOTE(sampling_strategy=.75, random_state=42)
X_train_log, y_train_log = sm.fit_sample(X_train_log, y_train_log)

# apply the Logistic Regression model
logistic_reg = ScratchLogisticRegression()
logistic_reg.train(X_train_log, y_train_log)

# check model against the validation set
y_pred_log_val = logistic_reg.predict(X_val_log)
accuracy = logistic_reg.accuracy(y_pred_log_val, y_val_log)
print('Prediction accuracy of validation set: %f' % accuracy)
log_conf_mtx = confusion_matrix(y_test_log, y_pred_log_val)
logreg_f1_score = f1_score(y_val_log, y_pred_log_val, average='weighted')
print('F1 Score of validation set: %f' % logreg_f1_score)
print('Confusion Matrix of validation set:')
print(log_conf_mtx)

# check model against the test set
# y_pred_log = logistic_reg.predict(X_test_log)
# accuracy = logistic_reg.accuracy(y_pred_log, y_test_log)
# print('Prediction accuracy of test set: %f' % accuracy)
# log_conf_mtx = confusion_matrix(y_test_log, y_pred_log)
# logreg_f1_score = f1_score(y_test_log, y_pred_log, average='weighted')
# print('F1 Score of test set: %f' % logreg_f1_score)
# print('Confusion Matrix of test set:')
# print(log_conf_mtx)
# print()

# # show plot of convergence of costs when calculating theta
# log_costs = logistic_reg.cal_costs()
# plt.plot(range(len(log_costs)), log_costs)
# plt.title('Convergence of cost function')
# plt.xlabel('iteration')
# plt.ylabel('cost')
# plt.show()
#


