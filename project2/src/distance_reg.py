import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd

water_df = pd.read_pickle("project2/pickled_data/water_dataset.pkl")

#log(energy) noise and a constant independent variables used for the regression
x = np.log(water_df["en_p"].values.reshape(-1,1))
noise = np.random.randn(len(x),1)
x_newmat = np.c_[ np.ones([len(x),1]),noise, x]
#dependent variable log(distance)
y = np.log(water_df["dist_p"].values)

#model = LinearRegression(fit_intercept=True)
model = LinearRegression()
model.fit(x,y)

r_sq = model.score(x, y)
print(f"coefficient of determination: {r_sq}")

print(f"intercept: {model.intercept_}")


print(f"slope: {model.coef_}")