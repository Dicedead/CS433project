import pandas as pd
import numpy as np


def augment_pickled_dataset(
        material: str,
        seed=12,
        verbose=False
) -> None:
    material_df = pd.read_pickle(f"../pickled_data/{material}_dataset.pkl")
    energy_levels = sorted(material_df["en_p"].unique())
    new_dfs = []
    num_samples = 9 * len(material_df) // len(energy_levels)

    for idx in range(len(energy_levels) - 1):
        curr_en = energy_levels[idx]
        next_en = energy_levels[idx + 1]
        df_curr_en = material_df[material_df["en_p"] == curr_en]
        df_next_en = material_df[material_df["en_p"] == next_en]

        curr_samples = df_curr_en.sample(n=num_samples, ignore_index=True, replace=True)
        new_samples = curr_samples.astype("float32").copy()
        next_dist_samples = df_next_en['dist_p'].sample(n=num_samples, ignore_index=True, replace=True)
        rs = np.random.RandomState(seed=seed)

        theta = np.random.random(num_samples)
        new_samples['en_p'] = (1 - theta) * curr_en + theta * next_en
        theta = np.random.random(num_samples)
        new_samples['dist_p'] = (1 - theta) * new_samples['dist_p'] + theta * next_dist_samples

        if verbose:
            print(f"Computed new samples between {curr_en} and {next_en}")

        new_dfs.append(new_samples)

    new_dfs.append(material_df.astype("float32"))
    pd.concat(new_dfs).to_pickle(f"../pickled_data/{material}_augmented_dataset.pkl")

    if verbose:
        print("Done!")


if __name__ == "__main__":
    augment_pickled_dataset("lung", verbose=True)
    # augment_pickled_dataset("water", verbose=True)
