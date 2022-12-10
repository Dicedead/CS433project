from sklearn import datasets
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


plt.style.use('ggplot')

#importing the data
lung_df = pd.read_pickle("project2/pickled_data/lung_dataset.pkl")

#energy independent variable used for the regression
x = lung_df["en_p"].values.reshape(-1,1)
#dependent variable distance
y = lung_df["dist_p"].values.reshape(-1,1)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3)
model = MLPRegressor(solver='adam', max_iter= 100,
                          learning_rate='adaptive', learning_rate_init=0.001, 
                         )
model.fit(x_train, y_train.ravel())
expected_y  = y_test
predicted_y = model.predict(x_test)
print(metrics.r2_score(expected_y, predicted_y))
print(metrics.mean_squared_log_error(expected_y, predicted_y))
print(metrics.mean_squared_error(expected_y, predicted_y))
