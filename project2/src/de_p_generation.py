import pandas as pd

if __name__ == "__main__":
    water_dataset = pd.read_pickle("../pickled_data/water_dataset.pkl")
    emissions = water_dataset[water_dataset["emission"] == 1]
    emissions.mean().to_pickle("../model_parameters/water/de_p_generation_dataset_stats/mean.pkl")
    emissions.std().to_pickle("../model_parameters/water/de_p_generation_dataset_stats/std.pkl")
