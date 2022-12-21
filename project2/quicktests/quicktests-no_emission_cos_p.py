import numpy as np

import cgan
from data_types import Type
from cos_parent_gan import *
import seaborn as sns
from scipy import stats

from ene_child_gan import EnergyChildGenerator

water_dataset = pd.read_pickle("../pickled_data/water_dataset.pkl")
no_emissions = water_dataset[water_dataset['emission'] == 0]

energies = no_emissions['en_p'].values
distances = no_emissions['dist_p'].values

def predict_no_emission_cos_p(
        p : Particle,
        distance: float,
        scale=1/80
):
    return np.clip(1-np.abs(np.random.laplace(0, np.minimum(scale, 1/(1 + p.ene * distance)))), -1, 1)# no_em_model.generate_from_particle(p, distance)[0]


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
