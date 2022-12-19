import numpy as np

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
ene_cs = emissions['en_c'].values

cos_p_model = CosParentGenerator.load("../model_parameters/water/cos_p_prediction.sav",
                                          "../model_parameters/water/cos_p_prediction_dataset_stats")

def predict_cos_p(p: Particle, distance: float, ene_c: float):
    return cos_p_model.generate_from_particle(p, distance, ene_c)[0]


values = np.array([predict_cos_p(Particle(0, 0, 0, 0, 0, 0, energies[x], Type.photon), distances[x], ene_cs[x])
                   for x in range(len(energies))])
sns.histplot(values)

# sns.histplot(emissions['en_c'])

# sns.histplot(emissions[['cos_p']])
# sns.histplot(emissions[['cos_c']])
# sns.histplot(emissions[['de_p']], log_scale=(False, True))
# sns.histplot(emissions[['en_c']])
# print(stats.spearmanr(emissions[['cos_p']], emissions[['cos_c']]))

plt.show()
