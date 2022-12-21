import numpy as np
import pandas as pd
import torch.distributions


def generate_de_p(
        de_p_mean: float,
        de_p_std: float,
        scale=1 / 100000
):
    return np.abs(np.random.laplace(de_p_mean, de_p_std * scale, 1))[0]


if __name__ == "__main__":
    water_dataset = pd.read_pickle("../pickled_data/water_dataset.pkl")
    emissions = water_dataset[water_dataset["emission"] == 1]
    emissions.mean().to_pickle("../model_parameters/water/de_p_generation_dataset_stats/mean.pkl")
    emissions.std().to_pickle("../model_parameters/water/de_p_generation_dataset_stats/std.pkl")
