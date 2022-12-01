import zipfile
import os
import pandas as pd

from jsontodf import jsontodf
from txtojson import txt_to_json


def material_dataset_txt_to_json(
        material: str,
        input_folder=None,
        output_folder=None
):
    if input_folder is None:
        input_folder = "../data/data_" + material + "/"

    if output_folder is None:
        output_folder = "../json_data/json_" + material + "/"

    for idx, file in enumerate(os.listdir(input_folder)):

        if file == ".gitkeep":
            continue

        with zipfile.ZipFile(input_folder + file, 'r') as zip_ref:
            zip_ref.extractall(input_folder)
        new_name = material + "_" + str(idx)
        txt_to_json("-", new_name + ".json",
                    verbose=True,
                    uncompressed=False,
                    separated_steps_emissions=True,
                    input_folder=input_folder,
                    output_folder=output_folder
                    )
        os.remove(input_folder + "-")


def material_dataset_jsontodf(
        material: str,
        json_folder=None,
        two_returns=False,
        verbose=False
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    if json_folder is None:
        json_folder = "../json_data/json_" + material + "/"

    emissions, no_emissions = [], []
    dataframes = []
    for file in os.listdir(json_folder):

        if file == ".gitkeep":
            continue

        if verbose:
            print(f"Converting {file}...")

        if two_returns:
            df1, df2 = jsontodf(
                input_json_filename=file,
                input_json_folder=json_folder
            )
            emissions.append(df1)
            no_emissions.append(df2)
        else:
            dataframes.append(jsontodf(
                input_json_filename=file,
                input_json_folder=json_folder
            ))

    if verbose:
        print(f"Concatenating dataframes")

    if two_returns:
        res = pd.concat(emissions), pd.concat(no_emissions)
    else:
        res = pd.concat(dataframes)

    if verbose:
        print("Done!")

    return res


if __name__ == "__main__":
    water_dataset = material_dataset_jsontodf("water", verbose=True, two_returns=False)
    water_dataset.to_pickle("../pickled_data/water_dataset.pkl")
