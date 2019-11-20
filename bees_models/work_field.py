import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models.lin_reg import LinearRegression as ScratchLinearRegression
from models.log_reg import LogisticRegression as ScratchLogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification



X, y = make_classification(n_samples=3000, n_features=2, n_redundant=0, n_informative=1,
                           n_clusters_per_class=1, random_state=14)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
logistic_reg = ScratchLogisticRegression()
logistic_reg.train(X_train, y_train)
y_pred = logistic_reg.predict(X_test)
accuracy = logistic_reg.accuracy(y_pred, y_test)
print('Prediction accuracy: %f' % accuracy)
