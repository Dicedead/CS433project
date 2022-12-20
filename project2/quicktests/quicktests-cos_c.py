import numpy as np

import cgan
from cos_child_gan import CosChildGenerator
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
cos_ps = emissions['cos_p'].values

cos_c_model = CosChildGenerator.load("../model_parameters/water/cos_c_prediction.sav",
                                          "../model_parameters/water/cos_c_prediction_dataset_stats")

def predict_cos_c(
        p: Particle,
        distance: float,
        ene_c: float,
        cos_p: float,
        loc=0.64,
        scale=0.6856
):
    pred = cos_c_model.generate_from_particle(p, distance, ene_c, cos_p)[0, 0]
    scale -= loc
    pred -= loc
    pred *= 1/scale
    return np.clip(pred, -1, 1)


values = np.array([predict_cos_c(Particle(0, 0, 0, 0, 0, 0, energies[x], Type.photon), distances[x], ene_cs[x], cos_ps[x])
                   for x in range(len(energies))])
print(values.max())
sns.histplot(values)

# sns.histplot(emissions['cos_c'])

# sns.histplot(emissions[['cos_p']])
# sns.histplot(emissions[['cos_c']])
# sns.histplot(emissions[['de_p']], log_scale=(False, True))
# sns.histplot(emissions[['en_c']])
# print(stats.spearmanr(emissions[['cos_p']], emissions[['cos_c']]))

plt.show()
