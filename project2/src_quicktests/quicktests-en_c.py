import numpy as np
import pandas as pd

import cgan
from data_types import Type
from cos_parent_gan import *
import seaborn as sns
from scipy import stats

from ene_child_gan import EnergyChildGenerator

water_dataset = pd.read_pickle("../pickled_data/water_dataset.pkl")
emissions = pd.read_pickle("../pickled_data/sampled_water_emissions.pkl")

energies = emissions['en_p'].values
distances = emissions['dist_p'].values

en_c_model = EnergyChildGenerator.load("../model_parameters/water/ene_c_prediction.sav",
                                           "../model_parameters/water/ene_c_prediction_dataset_stats")

def predict_en_c(
        p: Particle,
        distance: float,
        scaling_mean=1.515,
        scaling_std=5.75/0.45,
        max_iters=15
):
    it = 0
    while it < max_iters:
        pred = scaling_std * (en_c_model.generate_from_particle(p, distance)[0, 0] - scaling_mean)
        it += 1
        if pred >= 0:
            return pred
    return 0.


values = np.array([predict_en_c(Particle(0, 0, 0, 0, 0, 0, energies[x], Type.photon), distances[x])
                   for x in range(len(energies))])

sns.histplot(values)

# sns.histplot(emissions['en_c'])
# print(emissions['en_c'].mean())

# sns.histplot(emissions[['cos_p']])
# sns.histplot(emissions[['cos_c']])
# sns.histplot(emissions[['de_p']], log_scale=(False, True))
# sns.histplot(emissions[['en_c']])
# print(stats.spearmanr(emissions[['cos_p']], emissions[['cos_c']]))

plt.show()
