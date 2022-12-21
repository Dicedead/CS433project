import pickle

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

from data_types import Particle

__POLY_DEGREE = 3

def predict_emission(clf: LogisticRegression, p: Particle, distance: float):
    def __log_reg_features(dist_p, en_p):  # copied for efficiency purposes
        polyfeat = PolynomialFeatures(degree=__POLY_DEGREE)
        x = np.array([[dist_p, en_p]])
        y = polyfeat.fit_transform(x)
        t = np.exp(-np.array([[en_p, dist_p]]))
        return np.concatenate([x, y, t], axis=1)

    z = __log_reg_features(distance, p.ene)
    return clf.predict(z)[0]

def log_reg_features(df):
    polyfeat = PolynomialFeatures(degree=__POLY_DEGREE)
    return np.concatenate([df[["dist_p", "en_p"]],
                           polyfeat.fit_transform(df[["dist_p", "en_p"]]),
                           df[["en_p"]].apply(lambda x: np.exp(-x)),
                           df[["dist_p"]].apply(lambda x: np.exp(-x))], axis=1
                          )


if __name__ == "__main__":
    water_dataset = pd.read_pickle("../pickled_data/water_dataset.pkl")
    water_dataset = water_dataset
    training_ratio = 0.6
    training_set, test_set = train_test_split(water_dataset[["dist_p", "en_p", "emission"]], train_size=training_ratio)

    clf_logreg = LogisticRegression(class_weight="balanced")
    clf_logreg.fit(log_reg_features(training_set), training_set["emission"])

    filename = '../model_parameters/water/emission_prediction.sav'
    with open(filename, 'wb') as f:
        pickle.dump(clf_logreg, f)
