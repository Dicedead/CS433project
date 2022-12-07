import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import numpy as np


#importing the data
lung_df = pd.read_pickle("project2/pickled_data/lung_dataset.pkl")

#energy independent variable used for the regression
x = lung_df["en_p"].values.reshape(-1,1)
#dependent variable distance
y = lung_df["dist_p"].values.reshape(-1,1)

#feature scaling
sc_X = StandardScaler() 
sc_Y = StandardScaler() 
x = sc_X.fit_transform(x)
y = sc_X.fit_transform(y)



#Fitting the support vector regression model
regressor = SVR(kernel='rbf')
regressor.fit(x,y.ravel())


#See result
X_grid = np.arange(min(x), max(x), 0.1)
print(len(X_grid))
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Support Vector Regression Model')
plt.xlabel('Energy')
plt.ylabel('Distance')
plt.show()

