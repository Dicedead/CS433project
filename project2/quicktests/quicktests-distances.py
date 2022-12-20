import seaborn as sns

from data_types import Type
from distance_gan import *

water_dataset = pd.read_pickle("../pickled_data/water_dataset.pkl")
emissions = pd.read_pickle("../pickled_data/sampled_water_emissions.pkl")
energies = emissions[['en_p']]
# ene_scaler = StandardScaler()
# dist_scaler = StandardScaler()
#
#
# def predict_distance(
#         p: Particle,
#         scaling_mean: float = 250,
#         scaling_std: float = 1000 / 70
# ):
#     while True:
#         pred = scaling_std * (distance_model.generate_from_particle(p)[0] - scaling_mean)
#         if pred >= 0:
#             return pred
#
# distance_model = DistanceGenerator.load("../model_parameters/water/distance_prediction.sav",
#                                         "../model_parameters/water/distance_prediction_dataset_stats")
# values = np.array([predict_distance(Particle(0, 0, 0, 0, 0, 0, x, Type.photon))
#                    for x in energies.values]).squeeze()
# sns.histplot(values)
# print(emissions['dist_p'].mean())
sns.histplot(water_dataset[['dist_p']])

# print(distance_model(torch.distributions.Exponential(1/216).sample((len(energies[:30]), 2)),
#                             torch.from_numpy(
#     ene_scaler.fit_transform(energies[:30].values)).float()).detach().cpu()
#                             .numpy())

plt.show()
