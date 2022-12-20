import numpy.random
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

water_dataset = pd.read_pickle("../pickled_data/water_dataset.pkl")
emissions = water_dataset[water_dataset["emission"] == 1]


def generate_de_p(mean, std, size):
    return np.abs(np.random.laplace(mean, std, size))


de_ps = water_dataset['de_p']

sns.histplot(generate_de_p(de_ps.mean(), de_ps.std() / 1000000, len(de_ps)))
# sns.histplot(emissions['de_p'], log_scale=(False, True))
plt.show()
