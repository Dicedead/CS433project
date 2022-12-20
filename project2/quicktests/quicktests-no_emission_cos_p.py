import numpy as np

import cgan
from data_types import Type
from cos_parent_gan import *
import seaborn as sns
from scipy import stats

from ene_child_gan import EnergyChildGenerator
from no_emission_cos_p_gan import NoEmissionCosParentGenerator

water_dataset = pd.read_pickle("../pickled_data/water_dataset.pkl")
no_emissions = water_dataset[water_dataset['emission'] == 0]

energies = no_emissions['en_p'].values
distances = no_emissions['dist_p'].values

no_em_model = NoEmissionCosParentGenerator.load("../model_parameters/water/no_em_cos_p_prediction.sav",
                                          "../model_parameters/water/no_em_cos_p_prediction_dataset_stats")

def predict_no_emission_cos_p(p : Particle, distance: float):
    return 1-np.abs(np.random.laplace(0, np.minimum(1/80, 1/(p.ene * distance)))) # no_em_model.generate_from_particle(p, distance)[0]


values = np.array([predict_no_emission_cos_p(Particle(0, 0, 0, 0, 0, 0, energies[x], Type.photon), distances[x])
                    for x in range(len(energies))])
sns.histplot(values, log_scale=(False, True))

print(stats.spearmanr(no_emissions['dist_p'], values))
print(stats.spearmanr(no_emissions['en_p'], values))

print(stats.spearmanr(no_emissions['dist_p'], no_emissions['cos_p']))
print(stats.spearmanr(no_emissions['en_p'], no_emissions['cos_p']))

sns.histplot(no_emissions['cos_p'],  log_scale=(False, True))

# sns.histplot(emissions[['cos_p']])
# sns.histplot(emissions[['cos_c']])
# sns.histplot(emissions[['de_p']], log_scale=(False, True))
# sns.histplot(emissions[['en_c']])
# print(stats.spearmanr(emissions[['cos_p']], emissions[['cos_c']]))

plt.show()
