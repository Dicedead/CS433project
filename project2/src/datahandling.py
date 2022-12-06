import zipfile
import os
import pandas as pd

from jsontodf import jsontodf
from txtojson import txt_to_json


def material_dataset_txt_to_json(
        material: str,
        input_folder=None,
        output_folder=None,
        verbose=True
):
    if input_folder is None:
        input_folder = "../data/data_" + material + "/"

    if output_folder is None:
        output_folder = "../json_data/json_" + material + "/"

    for idx, file in enumerate(os.listdir(input_folder)):

        if file == ".gitkeep":
            continue
        if file == "-":
            os.remove(input_folder + file)
            continue

        if verbose:
            print(f"Converting {file} to json.")

        with zipfile.ZipFile(input_folder + file, 'r') as zip_ref:
            extracted_file_name = zip_ref.namelist()[0]
            zip_ref.extractall(input_folder)
        new_name = material + "_" + file
        txt_to_json(extracted_file_name, new_name + ".json",
                    verbose=verbose,
                    uncompressed=False,
                    separated_steps_emissions=True,
                    input_folder=input_folder,
                    output_folder=output_folder
                    )
        os.remove(input_folder + extracted_file_name)


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
            print(f"Converting {file} to dataframe.")

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
        print(f"Done! ({material})")

    return res


def from_zip_to_pickle(
        material: str,
        verbose=True,
):
    material_dataset_txt_to_json(material, verbose=verbose)
    material_dataset = material_dataset_jsontodf(material, verbose=verbose, two_returns=False)
    material_dataset.to_pickle(f"../pickled_data/{material}_dataset.pkl")


if __name__ == "__main__":
    # from_zip_to_pickle("water")
    from_zip_to_pickle("lung")
